#include "function/quantize.h"
#include "util/data.h"

// directly modify camArray to add quantization
void Quantize::write(CAMArrayBase* camArray){
    camArray->getType();
    // throw an not implemented error
    throw std::runtime_error("Quantize::write() not implemented");
}