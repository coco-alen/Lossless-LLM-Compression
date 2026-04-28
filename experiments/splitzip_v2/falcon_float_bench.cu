#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Falcon_float_pipeline.cuh"

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error in " << what << ": " << cudaGetErrorString(status) << "\n";
        std::exit(1);
    }
}

struct Args {
    std::string input;
    size_t num_elements = 0;
    int repeats = 10;
    int device = 0;
    size_t chunk_size = 0;
    int streams = 16;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto require_value = [&](const char* name) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--input") {
            args.input = require_value("--input");
        } else if (key == "--num-elements") {
            args.num_elements = std::stoull(require_value("--num-elements"));
        } else if (key == "--repeats") {
            args.repeats = std::stoi(require_value("--repeats"));
        } else if (key == "--device") {
            args.device = std::stoi(require_value("--device"));
        } else if (key == "--chunk-size") {
            args.chunk_size = std::stoull(require_value("--chunk-size"));
        } else if (key == "--streams") {
            args.streams = std::stoi(require_value("--streams"));
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.input.empty() || args.num_elements == 0) {
        throw std::runtime_error("--input and --num-elements are required");
    }
    if (args.repeats <= 0 || args.streams <= 0) {
        throw std::runtime_error("--repeats and --streams must be positive");
    }
    return args;
}

size_t choose_chunk_size(size_t nb_ele, int streams) {
    size_t chunk_size = 1025;
    size_t temp = std::max<size_t>(1, nb_ele / streams);
    size_t available = 0;
    size_t total = 0;
    check_cuda(cudaMemGetInfo(&available, &total), "cudaMemGetInfo");
    size_t limit = available / (4 * static_cast<size_t>(streams) * sizeof(float) * 2);
    while (chunk_size <= limit && chunk_size <= temp) {
        chunk_size *= 2;
    }
    chunk_size /= 2;
    return std::max<size_t>(1024, chunk_size);
}

ProcessedData_32 load_data(const Args& args) {
    ProcessedData_32 data{};
    data.nbEle = args.num_elements;
    const size_t bytes = args.num_elements * sizeof(float);

    check_cuda(cudaHostAlloc(&data.oriData, bytes, cudaHostAllocDefault), "cudaHostAlloc oriData");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&data.cmpBytes), bytes, cudaHostAllocDefault), "cudaHostAlloc cmpBytes");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&data.cmpSize), sizeof(unsigned int), cudaHostAllocDefault), "cudaHostAlloc cmpSize");
    check_cuda(cudaHostAlloc(&data.decData, bytes, cudaHostAllocDefault), "cudaHostAlloc decData");

    std::ifstream fin(args.input, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("failed to open input: " + args.input);
    }
    fin.read(reinterpret_cast<char*>(data.oriData), static_cast<std::streamsize>(bytes));
    if (fin.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("input file is shorter than --num-elements");
    }
    return data;
}

void free_data(ProcessedData_32& data) {
    cleanup_data_32(data);
    if (data.cmpSize != nullptr) {
        check_cuda(cudaFreeHost(data.cmpSize), "cudaFreeHost cmpSize");
        data.cmpSize = nullptr;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        check_cuda(cudaSetDevice(args.device), "cudaSetDevice");
        ProcessedData_32 data = load_data(args);
        size_t chunk_size = args.chunk_size ? args.chunk_size : choose_chunk_size(data.nbEle, args.streams);

        FalconPipeline_32 pipeline(args.streams);
        const double raw_gb = static_cast<double>(data.nbEle * sizeof(float)) / 1e9;

        std::cout << "FALCON_CONFIG"
                  << " num_elements=" << data.nbEle
                  << " chunk_size=" << chunk_size
                  << " streams=" << args.streams
                  << " repeats=" << args.repeats
                  << "\n";

        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            auto comp_result = pipeline.executeCompressionPipeline(data, chunk_size, args.streams);
            check_cuda(cudaDeviceSynchronize(), "compress synchronize");
            auto decomp = pipeline.executeDecompressionPipeline(comp_result, data, args.streams, false);
            check_cuda(cudaDeviceSynchronize(), "decompress synchronize");

            bool ok = std::memcmp(data.oriData, data.decData, data.nbEle * sizeof(float)) == 0;
            const double compressed_bytes = static_cast<double>(comp_result.analysis.total_compressed_size);
            const double ratio_fp32 = (data.nbEle * sizeof(float)) / compressed_bytes;

            std::cout << "FALCON_RESULT"
                      << " repeat=" << repeat
                      << " ok=" << (ok ? 1 : 0)
                      << " compressed_bytes=" << static_cast<size_t>(compressed_bytes)
                      << " ratio_fp32=" << ratio_fp32
                      << " comp_ms=" << comp_result.analysis.comp_time
                      << " decomp_ms=" << decomp.decomp_time
                      << " comp_gbs=" << comp_result.analysis.comp_throughout
                      << " decomp_gbs=" << decomp.decomp_throughout
                      << " raw_gb=" << raw_gb
                      << "\n";
        }

        free_data(data);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "FALCON_ERROR " << exc.what() << "\n";
        return 1;
    }
}
