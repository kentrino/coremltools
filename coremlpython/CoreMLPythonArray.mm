#include <iostream>
#import "CoreMLPythonArray.h"

using std::cout;
using std::endl;

@implementation PybindCompatibleArray

+ (MLMultiArrayDataType)dataTypeOf:(py::array)array {
    const auto& dt = array.dtype();
    char kind = dt.kind();
    size_t itemsize = dt.itemsize();
    cout << "itemsize: " << itemsize << ", kind: " << kind << endl;
    if(kind == 'i' && itemsize == 4) {
        return MLMultiArrayDataTypeInt32;
    } else if(kind == 'f' && itemsize == 4) {
        return MLMultiArrayDataTypeFloat32;
    } else if( (kind == 'f' || kind == 'd') && itemsize == 8) {
        return MLMultiArrayDataTypeDouble;
    }
    
    throw std::runtime_error("Unsupported array type: " + std::to_string(kind) + " with itemsize = " + std::to_string(itemsize));
}

+ (NSArray<NSNumber *> *)shapeOf:(py::array)array {
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        [ret addObject:[NSNumber numberWithUnsignedLongLong:array.shape(i)]];
    }
    return ret;
}

+ (NSArray<NSNumber *> *)stridesOf:(py::array)array {
    // numpy strides is in bytes.
    // this type must return number of ELEMENTS! (as per mlkit)
    
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        size_t stride = array.strides(i) / array.itemsize();
        [ret addObject:[NSNumber numberWithUnsignedLongLong:stride]];
    }
    return ret;
}

- (void) show {
    NSArray *shape = [self shape];
    int channel = 3;
    int width = [shape[1] intValue];
    int height = [shape[2] intValue];
    // 3, 500, 888
    NSLog(@"%@, %@, %@", shape[0], shape[1], shape[2]);
    // 3 * 500 * 888
    cout << "count:" << [self count] << endl;
    cout << "dataType:" << [self dataType] << endl;

    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
            NSNumber* nsW = [NSNumber numberWithInt:w];
            NSNumber* nsH = [NSNumber numberWithInt:h];
            NSArray *c0 = @[@0, nsH, nsW];
            NSNumber* nsValueC0 = [self objectForKeyedSubscript:c0];
            NSArray *c1 = @[@1, nsH, nsW];
            NSNumber* nsValueC1 = [self objectForKeyedSubscript:c1];
            NSArray *c2 = @[@2, nsH, nsW];
            NSNumber* nsValueC2 = [self objectForKeyedSubscript:c2];
            NSLog(@"[%d, %d] %@, %@, %@", h, w, nsValueC0, nsValueC1, nsValueC2);
        }
    }
}

- (PybindCompatibleArray *)initWithArray:(py::array)array {

    self = [super initWithDataPointer:array.mutable_data()
                                shape:[self.class shapeOf:array]
                             dataType:[self.class dataTypeOf:array]
                              strides:[self.class stridesOf:array]
                          deallocator:nil
                                error:nil];
    // [self show];

    if (self) {
        m_array = array;
    }
    return self;
}

@end
