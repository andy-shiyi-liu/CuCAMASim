#include "function/cuda/writeNoise.cuh"
#include "function/cuda/rram.cuh"


void addRRAMNoise(WriteNoise *writeNoise, ACAMArray *array){
    assert(writeNoise->getNoiseConfig()->device == "RRAM");
    assert(writeNoise->getHasNoise());




    std::cerr
      << "\033[33mWARNING: addRRAMNoise() is still under development"
      << std::endl;
}