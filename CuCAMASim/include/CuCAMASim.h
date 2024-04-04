#ifndef CUCAMASIM_H
#define CUCAMASIM_H

#include <iostream>
#include "util/config.h"

class CuCAMASim{
    private:
    const CamConfig *config;
    public:
    CuCAMASim(CamConfig *camConfig);
};
#endif