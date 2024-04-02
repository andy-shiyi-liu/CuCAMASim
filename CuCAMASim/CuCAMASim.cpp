#include <iostream>
#include "CuCAMASim.h"

CuCAMASim::CuCAMASim(){
    std::cout << "in CuCAMASim" << std::endl;
    camConfig camConfig("this is cam config path");
}