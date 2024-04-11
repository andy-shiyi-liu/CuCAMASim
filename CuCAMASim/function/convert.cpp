#include "function/convert.h"

#include "util/data.h"

// Converts data to a physical representation suitable for write operations.
// Depending on the CAM cell type (e.g., ACAM), it converts data to a physical
// voltage representation.
void ConvertToPhys::write(CAMData *camData) { camData->at(0, 0, 0); }