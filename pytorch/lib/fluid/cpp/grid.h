#include "torch/extension.h"

namespace fluid {

typedef at::Tensor T;

float getDx(at::Tensor self);

T interpol(const T& self, const T& pos);

void interpol1DWithFluid(
    const T& val_a, const T& is_fluid_a,
    const T& val_b, const T& is_fluid_b,
    const T& t_a, const T& t_b,
    T& is_fluid_ab, T& val_ab);

T interpolWithFluid(const T& self, const T& flags, const T& pos);

T getCentered(const T& self);
T getCentered_temp(const T& self);

T getAtMACX(const T& self);
T getAtMACY(const T& self);
T getAtMACZ(const T& self);

T getAtMACX_temp(const T& self);
T getAtMACY_temp(const T& self);
T getAtMACZ_temp(const T& self);

T interpolComponent(const T& self, const T& pos, int c);
T interpolComponent_temp(const T& self, const T& pos, int c);

T curl(const T& self);

} // namespace fluid
