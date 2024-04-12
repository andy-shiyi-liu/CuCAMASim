#include "function/quantize.h"
#include "util/data.h"

// directly modify camData to add quantization
void Quantize::write(CAMArray* camData){
    camData->getType();
    // throw an not implemented error
    throw std::runtime_error("Quantize::write() not implemented");
}