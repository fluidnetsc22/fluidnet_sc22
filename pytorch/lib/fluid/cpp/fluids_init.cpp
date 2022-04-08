#include "fluids_init.h"
#include <torch/torch.h>
#include <iostream>
#include <chrono>

namespace fluid {

typedef at::Tensor T;

// ****************************************************************************
// Advect Scalar
// ****************************************************************************

// Euler advection with line trace (as in Fluid Net)
T SemiLagrangeEulerFluidNet
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  T& i, T& j, T& k,
  const bool line_trace,
  const bool sample_outside_fluid
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  auto options = src.options();

  T ret = zeros_like(src);
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);

  //AT_ASSERTM(maskSolid.equal(1-maskFluid), "Masks are not complementary!");
  // Don't advect solid geometry.
  ret.masked_scatter_(maskSolid, src.masked_select(maskSolid));

  T pos = at::zeros({bsz, 3, d, h, w}, options).toType(src.scalar_type());

  pos.select(1,0) = i.toType(src.scalar_type()) + 0.5;
  pos.select(1,1) = j.toType(src.scalar_type()) + 0.5;
  pos.select(1,2) = k.toType(src.scalar_type()) + 0.5;

  T displacement = zeros_like(pos);

  // getCentered already eliminates border cells, no need to perform a masked select.
  displacement.masked_scatter_(maskBorder.ne(1), getCentered(vel));
  //displacement.masked_scatter_(maskBorder.ne(1), getCentered_temp(vel));

  displacement.mul_(-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  T back_pos = at::empty_like(pos);
  calcLineTrace(pos, displacement, flags, back_pos,line_trace);

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    ret.masked_scatter_(maskFluid,
         interpolWithFluid(src, flags, back_pos).masked_select(maskFluid));
  } else {
    ret.masked_scatter_(maskFluid,
         interpol(src, back_pos).masked_select(maskFluid));
  }
  return ret;
}

// Same kernel as previous one, except that it saves the
// particle trace position. This is used for the Fluid Net
// MacCormack routine (it does
// a local search around these positions in clamp routine).
T SemiLagrangeEulerFluidNetSavePos
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  T& i, T& j, T& k,
  const bool line_trace,
  const bool sample_outside_fluid,
  T& pos
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);

  auto options = src.options();

  T ret = zeros_like(src);
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);

  T start_pos = at::zeros({bsz, 3, d, h, w}, options).toType(src.scalar_type());

  start_pos.select(1,0) = i.toType(src.scalar_type()) + 0.5;
  start_pos.select(1,1) = j.toType(src.scalar_type()) + 0.5;
  start_pos.select(1,2) = k.toType(src.scalar_type()) + 0.5;

  T displacement = zeros_like(start_pos);

  // getCentered already eliminates border cells, no need to perform a masked select.
  displacement.masked_scatter_(maskBorder.ne(1), (-dt) * getCentered(vel));

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  T back_pos;
  calcLineTrace(start_pos, displacement, flags, back_pos, line_trace);

  pos.select(1,0).masked_scatter_(maskFluid.squeeze(1) , back_pos.select(1,0).masked_select(maskFluid.squeeze(1)));
  pos.select(1,1).masked_scatter_(maskFluid.squeeze(1), back_pos.select(1,1).masked_select(maskFluid.squeeze(1)));
  if (is3D) {
    pos.select(1,2).masked_scatter_(maskFluid.squeeze(1), back_pos.select(1,2).masked_select(maskFluid.squeeze(1)));
  }

  // Don't advect solid geometry.
  pos.select(1,0).masked_scatter_(maskSolid.squeeze(1), start_pos.select(1,0).masked_select(maskSolid.squeeze(1)));
  pos.select(1,1).masked_scatter_(maskSolid.squeeze(1), start_pos.select(1,1).masked_select(maskSolid.squeeze(1)));
  if (is3D) {
     pos.select(1,2).masked_scatter_(maskSolid.squeeze(1), start_pos.select(1,2).masked_select(maskSolid.squeeze(1)));
  }

  ret.masked_scatter_(maskSolid, src.masked_select(maskSolid));

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    ret.masked_scatter_(maskFluid,
         interpolWithFluid(src, flags, back_pos).masked_select(maskFluid));
  } else {
    ret.masked_scatter_(maskFluid,
         interpol(src, back_pos).masked_select(maskFluid));
  }
  return ret;
}

T MacCormackCorrect
(
  T& flags, const T& old,
  const T& fwd, const T& bwd,
  const float strength,
  bool is_levelset
)
{
  T dst = fwd.clone();
  T maskFluid = (flags.eq(TypeFluid));
  dst.masked_scatter_(maskFluid, (dst + strength * 0.5f * (old - bwd)).masked_select(maskFluid));

  return dst;
}

// Clamp routines. It is a search around a the input position
// position for min and max values. If no valid values are found, then
// false (in the mask) is returned (indicating that a clamp shouldn't be performed)
// otherwise true is returned (and the clamp min and max bounds are set).
T getClampBounds
(
  const T& src, const T& pos, const T& flags,
  const bool sample_outside,
  T& clamp_min, T& clamp_max
)
{
  int bsz = flags.size(0);
  int d   = flags.size(2) ;
  int h   = flags.size(3);
  int w   = flags.size(4);

  auto options = src.options();

  T minv = full_like(flags.toType(src.scalar_type()), INFINITY).squeeze(1);
  T maxv = full_like(flags.toType(src.scalar_type()), -INFINITY).squeeze(1);

  T i0 = at::zeros({bsz, d, h, w}, options).toType(at::kLong);
  T j0 = at::zeros({bsz, d, h, w}, options).toType(at::kLong);
  T k0 = at::zeros({bsz, d, h, w}, options).toType(at::kLong);

  i0 = clamp(pos.select(1,0).toType(at::kLong), 0, flags.size(4) - 1);
  j0 = clamp(pos.select(1,1).toType(at::kLong), 0, flags.size(3) - 1);
  k0 = (src.size(1) > 1) ?
      clamp(pos.select(1,2).toType(at::kLong), 0, flags.size(2) - 1) : zeros_like(i0);

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(i0.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  // We have to look all neighboors.
  T maskOutsideBounds = zeros_like(flags);
  T ncells = zeros_like(flags).squeeze(1);

  T i = zeros_like(i0);
  T j = zeros_like(j0);
  T k = zeros_like(k0);
  T zero = zeros_like(i0);

  for (int32_t dk = -1; dk <= 1; dk++) {
    for (int32_t dj = -1; dj <= 1; dj++) {
      for (int32_t di = -1; di<= 1; di++) {
        maskOutsideBounds = (( (k0 + dk) < 0).__or__( (k0 + dk) >= flags.size(2)).__or__
                             ( (j0 + dj) < 0).__or__( (j0 + dj) >= flags.size(3)).__or__
                             ( (i0 + di) < 0).__or__( (i0 + di) >= flags.size(4)));

        i = zero.where( (i0 + di < 0).__or__(i0 + di >= flags.size(4)), i0 + di);
        j = zero.where( (j0 + dj < 0).__or__(j0 + dj >= flags.size(3)), j0 + dj);
        k = zero.where( (k0 + dk < 0).__or__(k0 + dk >= flags.size(2)), k0 + dk);

        T flags_ijk = flags.index({idx_b, zero, k, j, i});
        T src_ijk = src.index({idx_b, zero, k, j, i});
        T maskSample = maskOutsideBounds.ne(1).__and__(flags_ijk.eq(TypeFluid).__or__(sample_outside));

        minv.masked_scatter_(maskSample, at::min(minv, src_ijk).masked_select(maskSample));
        maxv.masked_scatter_(maskSample, at::max(maxv, src_ijk).masked_select(maskSample));
        ncells.masked_scatter_(maskSample, (ncells + 1).masked_select(maskSample));
      }
    }
  }

  T ret = zeros_like(flags).toType(at::kByte);
  ncells = ncells.unsqueeze(1);
  clamp_min.masked_scatter_( (ncells >= 1) , minv.unsqueeze(1).masked_select( ncells >= 1));
  clamp_max.masked_scatter_( (ncells >= 1) , maxv.unsqueeze(1).masked_select( ncells >= 1));
  ret.masked_fill_( (ncells >= 1), 1);

  return ret;
}

T MacCormackClampFluidNet(
  T& flags, T& vel,
  const T& dst, const T& src,
  const T& fwd, const T& fwd_pos,
  const T& bwd_pos, const bool sample_outside
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);

  auto options = src.options();

  // Calculate the clamp bounds.
  T clamp_min = full_like(src, INFINITY);
  T clamp_max = full_like(src, -INFINITY);

  // Calculate the clamp bounds around the forward position.
  T pos = at::zeros({bsz, 3, d, h, w}, options).toType(fwd_pos.scalar_type());
  pos.select(1,0) = fwd_pos.select(1,0);
  pos.select(1,1) = fwd_pos.select(1,1);
  if (is3D) {
    pos.select(1,2) = fwd_pos.select(1,2);
  }

  T do_clamp_fwd = getClampBounds(
    src, pos, flags, sample_outside, clamp_min, clamp_max);

  // According to Selle et al. (An Unconditionally Stable MacCormack Method) only
  // a forward search is necessary.

  // do_clamp_fwd = false: If the cell is surrounded by fluid neighbors either
  // in the fwd or backward directions, then we need to revert to an euler step.
  // Otherwise, we found valid values with which to clamp the maccormack corrected
  // quantity. Apply this clamp.

  return fwd.where(do_clamp_fwd.ne(1), at::max( clamp_min, at::min(clamp_max, dst)));
}

at::Tensor advectScalar
(
  float dt, T src, T U, T flags,
  const std::string method_str,
  int bnd,
  const bool sample_outside_fluid,
  const float maccormack_strength
) {
  // Size checking done in python side
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);

  T s_dst = at::zeros_like(src);

  T fwd = at::zeros_like(src);
  T bwd = at::zeros_like(src);
  T fwd_pos = at::zeros_like(U);
  T bwd_pos = at::zeros_like(U);

  auto options = src.options();

  AdvectMethod method = StringToAdvectMethod(method_str);
  const bool is_levelset = false;
  const int order_space = 1;
  const bool line_trace = false;

  T pos_corrected = at::zeros({bsz, 3, d, h, w}, options).toType(src.scalar_type());

  T cur_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ? fwd : s_dst;

  T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T idx_y = at::arange(0, h, options).view({1,h,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(0, d, options).view({1,d,1,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
  }

  //T maskBorder = flags.eq(TypeObstacle);


  T maskBorder = (idx_x < bnd).__or__
                 (idx_x > w - 1 - bnd).__or__
                 (idx_y < bnd).__or__
                 (idx_y > h - 1 - bnd);

  if (is3D) {
      maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                    (idx_z > d - 1 - bnd);
  }
  maskBorder.unsqueeze_(1);

  // Manta zeros stuff on the border.
  cur_dst.masked_fill_(maskBorder, 0);

  pos_corrected.select(1,0) = idx_x.toType(src.scalar_type()) + 0.5;
  pos_corrected.select(1,1) = idx_y.toType(src.scalar_type()) + 0.5;
  pos_corrected.select(1,2) = idx_z.toType(src.scalar_type()) + 0.5;


  fwd_pos.select(1,0).masked_scatter_(maskBorder.squeeze(1), pos_corrected.select(1,0).masked_select(maskBorder.squeeze(1)));
  fwd_pos.select(1,1).masked_scatter_(maskBorder.squeeze(1), pos_corrected.select(1,1).masked_select(maskBorder.squeeze(1)));
  if (is3D) {
    fwd_pos.select(1,2).masked_scatter_(maskBorder.squeeze(1), pos_corrected.select(1,2).masked_select(maskBorder.squeeze(1)));
  }


  // Forward step.
  T val = at::empty_like(src);
  if (method == ADVECT_EULER_FLUIDNET) {
    val = SemiLagrangeEulerFluidNet(flags, U, src, maskBorder, dt, order_space,
            idx_x, idx_y, idx_z, line_trace, sample_outside_fluid);
  } else if (method == ADVECT_MACCORMACK_FLUIDNET) {
    val = SemiLagrangeEulerFluidNetSavePos(flags, U, src, maskBorder, dt, order_space,
            idx_x, idx_y, idx_z, line_trace, sample_outside_fluid, fwd_pos);
  } else {
    AT_ERROR("Advection method not supported!");
  }
  cur_dst.masked_scatter_(maskBorder.eq(0), val.masked_select(maskBorder.eq(0)));


  if (method != ADVECT_MACCORMACK_FLUIDNET) {
    // We're done. The forward Euler step is already in the output array.
    s_dst = cur_dst;
  } else {
    // Otherwise we need to do the backwards step (which is a SemiLagrange
    // step on the forward data - hence we need to finish the above ops
    // before moving on).)
    // Manta zeros stuff on the border.
    bwd.masked_fill_(maskBorder, 0);
    pos_corrected.select(1,0) = idx_x.toType(src.scalar_type())+ 0.5;
    pos_corrected.select(1,1) = idx_y.toType(src.scalar_type())+ 0.5;
    pos_corrected.select(1,2) = idx_z.toType(src.scalar_type())+ 0.5;

    bwd_pos.masked_scatter_(maskBorder, pos_corrected.masked_select(maskBorder));

    // Backwards step
    if (method == ADVECT_MACCORMACK_FLUIDNET) {
      bwd.masked_scatter_(maskBorder.ne(1),
          SemiLagrangeEulerFluidNetSavePos(flags, U, fwd, maskBorder, -dt, order_space,
          idx_x, idx_y, idx_z, line_trace, sample_outside_fluid, bwd_pos)
          .masked_select(maskBorder.ne(1)));
    }
    // Now compute the correction.
    s_dst = MacCormackCorrect(flags, src, fwd, bwd, maccormack_strength, is_levelset);

    // Now perform the clamping.
    if (method == ADVECT_MACCORMACK_FLUIDNET) {
      s_dst.masked_scatter_(maskBorder.ne(1),
          MacCormackClampFluidNet(flags, U, s_dst, src, fwd, fwd_pos, bwd_pos,
          sample_outside_fluid).masked_select(maskBorder.ne(1)));
    }
  }

  return s_dst;
}

// ****************************************************************************
// Advect Velocity
// ***************************************************************************

T SemiLagrangeEulerFluidNetMAC
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  const bool line_trace,
  T& i, T& j, T& k
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);

  auto options = src.options();

  T zero = zeros_like(src);
  T ret = at::zeros({bsz,3,d,h,w}, options).toType(src.scalar_type());
  T vec3_0 = at::zeros({bsz,3,d,h,w}, options).toType(src.scalar_type());
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);

  // Don't advect solid geometry.
  ret.select(1,0).unsqueeze(1).masked_scatter_(
          maskSolid, src.select(1,0).unsqueeze(1).masked_select(maskSolid));
  ret.select(1,1).unsqueeze(1).masked_scatter_(
          maskSolid, src.select(1,1).unsqueeze(1).masked_select(maskSolid));
  if (is3D) {
    ret.select(1,2).unsqueeze(1).masked_scatter_(
            maskSolid, src.select(1,2).unsqueeze(1).masked_select(maskSolid));
  }
  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  T pos = at::zeros({bsz, 3, d, h, w}, options).toType(src.scalar_type());

  pos.select(1,0) = i.toType(src.scalar_type()) + 0.5;
  pos.select(1,1) = j.toType(src.scalar_type()) + 0.5;
  pos.select(1,2) = k.toType(src.scalar_type()) + 0.5;

  // FluidNet: We floatly want to clamp to the SMALLEST of the steps in each
  // dimension, however this is OK for now (because doing so would expensive)...
  T xpos;
  calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
              getAtMACX(vel)) * (-dt), flags, xpos, line_trace);
  //calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
  //            getAtMACX_temp(vel)) * (-dt), flags, xpos, line_trace);


  // Two differnet interpolation functions!
  const T vx = interpolComponent(src, xpos, 0);
  //const T vx = interpolComponent_temp(src, xpos, 0);

  T ypos;

  calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
              getAtMACY(vel)) * (-dt), flags, ypos,line_trace);

  // Two differnet interpolation functions!
  const T vy = interpolComponent(src, ypos, 1);

  if (is3D) {
    T zpos;
    calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
                getAtMACZ(vel)) * (-dt), flags, zpos,line_trace);
    const T vz = interpolComponent(src, zpos, 2);

    ret.masked_scatter_(maskFluid, (at::cat({vx, vy, vz}, 1)).masked_select(maskFluid));
  }else{
    const T vz = zeros_like(vy);
    ret.masked_scatter_(maskFluid, (at::cat({vx, vy, vz}, 1)).masked_select(maskFluid));
  }


  return ret;
}

T MacCormackCorrectMAC
(
  T& flags, const T& old,
  const T& fwd, const T& bwd,
  const float strength,
  T& i, T& j, T& k
) {
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);

  auto options = old.options();

  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(at::kLong);
  idx_b = idx_b.expand({bsz,d,h,w});

  T skip = at::zeros({bsz, 3, d, h, w}, options).toType(at::kByte);

  T maskSolid = flags.ne(TypeFluid);
  skip.masked_fill_(maskSolid, 1);

  // This allows to never access negative indexes!
  T mask0 = zeroBy.where(i<=0, (flags.index({idx_b, zero, k, j, i-1}).ne(TypeFluid)));
  skip.select(1,0).masked_fill_(mask0, 1);

  T mask1 = zeroBy.where(j<=0, (flags.index({idx_b, zero, k, j-1, i}).ne(TypeFluid)));
  skip.select(1,1).masked_fill_(mask1, 1);

  if (is3D) {
    T mask2 = zeroBy.where(k<=0, (flags.index({idx_b, zero, k-1, j, i}).ne(TypeFluid)));
    skip.select(1,2).masked_fill_(mask2, 1);
  }

  T dst = at::zeros({bsz, (is3D? 3:2), d, h, w}, options).toType(flags.scalar_type());
  const int dim = is3D? 3 : 2;

  for (int c = 0; c < dim; ++c) {
    dst.select(1,c) = at::where(skip.select(1,c), fwd.select(1,c),
            fwd.select(1,c) + strength * 0.5f * (old.select(1,c) - bwd.select(1,c)));
  }
  return dst;
}

T doClampComponentMAC
(
  int chan,
  const T& flags, const T& dst,
  const T& orig,  const T& fwd,
  const T& pos, const T& vel
) {
  int bsz = flags.size(0);
  int d   = flags.size(2) ;
  int h   = flags.size(3);
  int w   = flags.size(4);
  bool is3D = (d > 1);

  auto options = dst.options();

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(at::kLong);
  idx_b = idx_b.expand({bsz,d,h,w});

  T ret = zeros_like(fwd);

  T minv = full_like(flags.toType(dst.scalar_type()), INFINITY);
  T maxv = full_like(flags.toType(dst.scalar_type()), -INFINITY);

  // forward and backward
  std::vector<T> positions;
  positions.insert(positions.end(), (pos - vel).toType(at::kInt));
  positions.insert(positions.end(), (pos + vel).toType(at::kInt));
  T maskRet = ones_like(flags).toType(at::kByte);

  for (int l = 0; l < 2; ++l) {
    T curr_pos = positions[l];

    // clamp forward lookup to grid
    T i0 = curr_pos.select(1,0).clamp(0, flags.size(4) - 2).toType(at::kLong);
    T j0 = curr_pos.select(1,1).clamp(0, flags.size(3) - 2).toType(at::kLong);
    T k0 = curr_pos.select(1,2).clamp(0,
                      is3D ? (flags.size(2) - 2) : 0).toType(at::kLong);

    T i1 = i0 + 1;
    T j1 = j0 + 1;
    T k1 = (is3D) ? (k0 + 1) : k0;

    int bnd = 0;
    T NotInBounds = (i0 < bnd).__or__
                    (i0 > w - 1 - bnd).__or__
                    (j0 < bnd).__or__
                    (j0 > h - 1 - bnd).__or__
                    (i1 < bnd).__or__
                    (i1 > w - 1 - bnd).__or__
                    (j1 < bnd).__or__
                    (j1 > h - 1 - bnd);
    if (is3D) {
        NotInBounds = NotInBounds.__or__(k0 < bnd).__or__
                                        (k0 > d - 1 - bnd).__or__
                                        (k1 < bnd).__or__
                                        (k1 > d - 1 - bnd);
    }
    // We make sure that we don't get out of bounds in call for index.
    // It does not matter the value we fill in, as long as it stays in bounds
    // (0 is straightforward), it will not be selected thanks to the mask InBounds.
    i0.masked_fill_(NotInBounds, 0);
    j0.masked_fill_(NotInBounds, 0);
    k0.masked_fill_(NotInBounds, 0);
    i1.masked_fill_(NotInBounds, 0);
    j1.masked_fill_(NotInBounds, 0);
    k1.masked_fill_(NotInBounds, 0);
    T c = at::zeros({1}, options).toType(idx_b.scalar_type());
    c[0] = chan;

    NotInBounds = NotInBounds.unsqueeze(1);
    T InBounds = NotInBounds.ne(1);

    ret.masked_scatter_(NotInBounds, fwd.masked_select(NotInBounds));
    maskRet.masked_fill_(NotInBounds, 0);

    // find min/max around source position
    T orig000 = orig.index({idx_b, c, k0, j0, i0}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig000).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig000).masked_select(InBounds));

    T orig100 = orig.index({idx_b, c, k0, j0, i1}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig100).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig100).masked_select(InBounds));

    T orig010 = orig.index({idx_b, c, k0, j1, i0}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig010).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig010).masked_select(InBounds));

    T orig110 = orig.index({idx_b, c, k0, j1, i1}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig110).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig110).masked_select(InBounds));

    if (is3D) {
      T orig001 = orig.index({idx_b, c, k1, j0, i0}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig001).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig001).masked_select(InBounds));

      T orig101 = orig.index({idx_b, c, k1, j0, i1}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig101).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig101).masked_select(InBounds));

      T orig011 = orig.index({idx_b, c, k1, j1, i0}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig011).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig011).masked_select(InBounds));

      T orig111 = orig.index({idx_b, c, k1, j1, i1}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig111).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig111).masked_select(InBounds));
    }
  }
  // clamp dst
  ret.masked_scatter_(maskRet,
          at::max(at::min(dst, maxv), minv).masked_select(maskRet));
  return ret;
}


T ApplyOutflow
(
    const T& flags, const T& vel,
    const T& velPrev,
    float dt,
    int bWidth
) {

    AT_ASSERTM(vel.dim() == 5 && flags.dim() == 5,
             "Dimension mismatch");
    AT_ASSERTM(flags.size(1) == 1, "flags is not scalar");

    int bsz = flags.size(0);
    int d   = flags.size(2);
    int h   = flags.size(3);
    int w   = flags.size(4);
    bool is3D = (d > 1);
    const int32_t bnd =1;
    int vlsz = vel.size(1);

    auto options = vel.options();
    int numel = d * h * w;

    T velDst = vel.clone();
    T ones_vel = at::ones_like(vel);;
    //////////////////////////////////////////////////////////////


    if (!is3D) {
    AT_ASSERTM(d == 1, "d > 1 for a 2D domain");
    }

    AT_ASSERTM(vel.is_contiguous() && flags.is_contiguous(), "Input is not contiguous");

    T vel_prev = vel.clone();
    //T vel_prev = at::zeros({bsz, 1, d, h, w}, options).toType(vel.scalar_type());

    T mCont = at::ones({bsz, 1, d, h, w}, options).toType(at::kByte); // Continue mask

    T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
    T idx_y = at::arange(0, h, options).view({1,h,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
    T idx_z = zeros_like(idx_x);
    if (is3D) {
       idx_z = at::arange(0, d, options).view({1,d,1,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
    }

    T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(at::kLong);
    idx_b = idx_b.expand({bsz,d,h,w});

    T maskBorder = (idx_x < bnd).__or__
                   (idx_x > w - 1 - bnd).__or__
                   (idx_y < bnd).__or__
                   (idx_y > h - 1 - bnd);
    if (is3D) {
        maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                      (idx_z > d - 1 - bnd);
    }
    maskBorder.unsqueeze_(1);

    //cur_vel->masked_fill_(maskBorder, 0);
    mCont.masked_fill_(maskBorder, 0);

    T maskFluid = flags.eq(TypeFluid);
    T maskObstacle = flags.eq(TypeObstacle).__and__(mCont);
    //cur_vel->masked_fill_(maskObstacle, 0);
    mCont.masked_fill_(maskObstacle, 0);

    T zero_f = at::zeros_like(vel); // Floating zero
    T zero_l = at::zeros_like(vel).toType(at::kLong); // Long zero (for index)
    T zero_l1 = at::zeros_like(flags).toType(at::kLong); // Long zero (for index)
    T one_l = at::ones_like(flags).toType(at::kLong); // Long zero (for index)
    T zeroBy = at::zeros_like(vel).toType(at::kByte); // Long zero (for index)
    T oneBy = at::ones_like(vel).toType(at::kByte); // Long zero (for index)
    // Otherwise, we are in a fluid or empty cell.
    // First, we get all the neighbors.

    T two_dim = at::arange(0, vlsz, options).view({1,vlsz,1,1,1}).toType(at::kLong);
    two_dim = two_dim.expand({bsz,vlsz,d,h,w});
    //T velC = *cur_vel_prev;

    T Outflow_Cont =  mCont.__or__(oneBy.
         where(flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeOutflow),zeroBy));

    T Outflow = oneBy.
         where(flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeOutflow),zeroBy);

    T i_l = zero_l.where( (idx_x <=0), idx_x - 1);
    T vel_1 = zero_f.
        where(Outflow_Cont.ne(1), (vel_prev).index({idx_b, two_dim, idx_z, idx_y, i_l}));

    T i_r = zero_l.where( (idx_x > w - 1 - bnd), idx_x + 1);
    T vel_2 = zero_f.
        where(Outflow_Cont.ne(1), (vel_prev).index({idx_b, two_dim, idx_z, idx_y, i_r}));

    T j_l = zero_l.where( (idx_y <= 0), idx_y - 1);
    T vel_3 = zero_f.
        where(Outflow_Cont.ne(1), (vel_prev).index({idx_b, two_dim, idx_z, j_l, idx_x}));

    T j_r = zero_l.where( (idx_y > h - 1 - bnd), idx_y + 1);
    T vel_4 = zero_f.
        where(Outflow_Cont.ne(1), (vel_prev).index({idx_b, two_dim, idx_z, j_r, idx_x}));

    T k_l = zero_l.where( (idx_z <= 0), idx_z - 1);
    T vel_5 = is3D ? zero_f.
        where(Outflow_Cont.ne(1), (vel_prev).index({idx_b, two_dim, k_l, idx_y, idx_x})) : zero_f;

    T k_r = zero_l.where( (idx_z > d - 1 - bnd), idx_z + 1);
    T vel_6 = is3D ? zero_f.
        where(Outflow_Cont.ne(1), (vel_prev).index({idx_b, two_dim, k_r, idx_y, idx_x})) : zero_f;


    T neighborLeftOut = (oneBy.where(flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeFluid),zeroBy)).__or__(
         oneBy.where(flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeOutflow),zeroBy));
    T neighborRightOut = (oneBy.where(flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeFluid),zeroBy)).__or__(
         oneBy.where(flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeOutflow),zeroBy));
    T neighborBotOut = (oneBy.where(flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeFluid),zeroBy)).__or__(
         oneBy.where(flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeOutflow),zeroBy));
    T neighborUpOut = (oneBy.where(flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeFluid),zeroBy)).__or__(
         oneBy.where(flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeOutflow),zeroBy));

    T neighborLeftObs = mCont.__and__(zeroBy.
                                      where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeObstacle)));
    T neighborRightObs = mCont.__and__(zeroBy.
                                       where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeObstacle)));
    T neighborBotObs = mCont.__and__(zeroBy.
                                        where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeObstacle)));
    T neighborUpObs = mCont.__and__(zeroBy.
                                    where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeObstacle)));


    T neighborBackOut = zeroBy;
    T neighborFrontOut = zeroBy;

    if (is3D) {

      T neighborBackOut = (oneBy.where(flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeFluid),zeroBy)).__or__(
           oneBy.where(flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeOutflow),zeroBy));
      T neighborFrontOut = (oneBy.where(flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeFluid),zeroBy)).__or__(
           oneBy.where(flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeOutflow),zeroBy));

    }


    T Bulk_Vel = 2*vel + vel_1 + vel_2 + vel_3 + vel_4;
    T Bulk_flags = 2* Outflow_Cont + neighborLeftOut + neighborRightOut + neighborBotOut + neighborUpOut;

    Bulk_Vel.masked_fill_(Outflow.ne(1), 0);
    Bulk_flags.masked_fill_(Outflow.ne(1), 0);

    // To avoid dividing by zero
    Bulk_flags.masked_fill_(Bulk_flags.eq(0), 1);

    T Final_vel = Bulk_Vel/Bulk_flags.toType(at::kFloat);


    velDst = vel_6;

    vel_1.masked_fill_(maskFluid.eq(1),0);
    vel_2.masked_fill_(maskFluid.eq(1),0);
    vel_3.masked_fill_(maskFluid.eq(1),0);
    vel_4.masked_fill_(maskFluid.eq(1),0);

    T vel_neigh_outflow = vel_1 + vel_2 + vel_3 + vel_4;

    // Add third dimension!!!
    // Et unlever la partie Ux dans les vel_neigh_outflow x et Uy pour les vel_neigh_outflow y

    T U_x_matrix = Outflow.__and__(oneBy.where(((flags.index({idx_b,  zero_l , idx_z, j_l, idx_x}).eq(TypeFluid).__or__
                      (flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeFluid))).__and__
                      (flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeOutflow))),zeroBy));

    T U_y_matrix = Outflow.__and__(oneBy.where(((flags.index({idx_b,  zero_l , idx_z, idx_y, i_l}).eq(TypeFluid).__or__
                      (flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeFluid))).__and__
                      (flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeOutflow))),zeroBy));

    T U_z_matrix = Outflow.__and__(oneBy.where(((flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeFluid).__or__
                      (flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeFluid))).__and__
                      (flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeOutflow))),zeroBy));


    T vel_neigh_outflow_x =  vel_neigh_outflow.clone();
    T vel_neigh_outflow_y =  vel_neigh_outflow.clone();

    vel_neigh_outflow_x.masked_fill_(U_x_matrix.eq(1),0);
    vel_neigh_outflow_y.masked_fill_(U_y_matrix.eq(1),0);

    T new_tensor_x = vel_neigh_outflow_x.transpose(0,1);
    T new_tensor_y = vel_neigh_outflow_y.transpose(0,1);
    T new_tensor_v = vel_neigh_outflow.transpose(0,1);


    new_tensor_v[0] = new_tensor_x[0];
    new_tensor_v[1] = new_tensor_y[1];


    T final_neigh_outflow = new_tensor_v.transpose(0,1);


    T maxMask = zeroBy.where(ones_vel < Final_vel, oneBy);
    T factor = ones_vel.masked_fill_(maxMask.eq(0),0) + Final_vel.masked_fill_(maxMask.eq(1),0);

    std::cout << "Factor  " <<  factor << std::endl;

    T velAux = ((vel - velPrev)/(factor))+final_neigh_outflow;

    velAux.masked_fill_(maskFluid.eq(1),0);
    velDst = velAux + vel;


    std::cout << "velDst  " <<  velDst << std::endl;

  /////////////////////////////////////////////////////////////

 return velDst;
}


T MacCormackClampMAC
(
  const T& flags, const T& vel, const T& dval,
  const T& orig, const T& fwd, const T& maskBorder,
  float dt,
  const T& i, const T& j, const T& k
) {

  int bsz = flags.size(0);
  int d   = flags.size(2);
  int h   = flags.size(3);
  int w   = flags.size(4);
  bool is3D = (d > 1);

  auto options = vel.options();

  T zero = at::zeros({bsz, 3, d, h, w}, options).toType(vel.scalar_type());
  T pos = at::cat({i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)}, 1).toType(vel.scalar_type());
  T dfwd = fwd.clone();

  // getAtMACX-Y-Z already eliminates border cells. In border cells we set 0 as vel
  // but it will be selected out by mask in advectVel.
  dval.select(1,0) = doClampComponentMAC(0, flags, dval.select(1,0).unsqueeze(1),
    orig,  dfwd.select(1,0).unsqueeze(1), pos,
    zero.masked_scatter_(maskBorder.eq(0), getAtMACX(vel)) * dt).squeeze(1);

  dval.select(1,1) = doClampComponentMAC(1, flags, dval.select(1,1).unsqueeze(1),
    orig,  dfwd.select(1,1).unsqueeze(1), pos,
   zero.masked_scatter_(maskBorder.eq(0), getAtMACY(vel)) * dt).squeeze(1);
  if (is3D) {
     dval.select(1,2) = doClampComponentMAC(2, flags, dval.select(1,2).unsqueeze(1),
        orig,  dfwd.select(1,2).unsqueeze(1), pos,
        zero.masked_scatter_(maskBorder.eq(0), getAtMACZ(vel)) * dt).squeeze(1);

  } else {
     dval.select(1,2).fill_(0);
  }
  return dval;
}

at::Tensor advectVel
(
  float dt, T orig, T U, T flags,
  const std::string method_str,
  int bnd,
  bool Openbound,
  const float maccormack_strength
) {
  // We treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.

  int bsz = flags.size(0);
  int d   = flags.size(2);
  int h   = flags.size(3);
  int w   = flags.size(4);

  bool is3D = (U.size(1) == 3);

  auto options = orig.options();

  T U_dst = zeros_like(U);

  // FluidNet did self-advection, but the introduction of viscosity
  // forces the advection of the viscous one by a non-divergent one
  // from previous time step.
  // T orig = U.clone();

  // The maccormack method also needs fwd and bwd temporary arrays.
  T fwd = at::zeros({bsz, U.size(1), d, h, w}, options).toType(flags.scalar_type());
  T bwd = at::zeros({bsz, U.size(1), d, h, w}, options).toType(flags.scalar_type());

  AdvectMethod method = StringToAdvectMethod(method_str);

  const int order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to FluidNet
  // methods only).
  const bool line_trace = false;

  T pos_corrected = at::zeros({bsz, 3, d, h, w}, options).toType(orig.scalar_type());

  T cur_U_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ? fwd : U_dst;

  T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T idx_y = at::arange(0, h, options).view({1,h,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
  T idx_z = zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(0, d, options).view({1,d,1,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
  }

  //T maskBorder = flags.eq(TypeObstacle);

  T maskBorder = (idx_x < bnd).__or__
                 (idx_x > w - 1 - bnd).__or__
                 (idx_y < bnd).__or__
                 (idx_y > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                    (idx_z > d - 1 - bnd);
  }
  maskBorder = maskBorder.unsqueeze(1);

  // Manta zeros stuff on the border.
  cur_U_dst.select(1,0).masked_scatter_(maskBorder.squeeze(1),
       pos_corrected.select(1,0).masked_select(maskBorder.squeeze(1)));
  cur_U_dst.select(1,1).masked_scatter_(maskBorder.squeeze(1),
       pos_corrected.select(1,1).masked_select(maskBorder.squeeze(1)));
  if (is3D) {
    cur_U_dst.select(1,2).masked_scatter_(maskBorder.squeeze(1),
         pos_corrected.select(1,2).masked_select(maskBorder.squeeze(1)));
  }
  // Forward step.
  T val;
  if (method == ADVECT_EULER_FLUIDNET ||
      method == ADVECT_MACCORMACK_FLUIDNET) {
    //std::cout << "First Euler  " << std::endl;
    val = SemiLagrangeEulerFluidNetMAC(flags, U, orig, maskBorder, dt, order_space,
            line_trace, idx_x, idx_y, idx_z);
    //std::cout << "End of First Euler  " << std::endl;
  } else {
    AT_ERROR("No defined method for MAC advection");
  }
  // Store in the output array.
  cur_U_dst.select(1,0).masked_scatter_(maskBorder.eq(0).squeeze(1),
       val.select(1,0).masked_select(maskBorder.eq(0).squeeze(1)));
  cur_U_dst.select(1,1).masked_scatter_(maskBorder.eq(0).squeeze(1),
       val.select(1,1).masked_select(maskBorder.eq(0).squeeze(1)));
  if (is3D) {
    cur_U_dst.select(1,2).masked_scatter_(maskBorder.eq(0).squeeze(1),
         val.select(1,2).masked_select(maskBorder.eq(0).squeeze(1)));
  }

  if (method != ADVECT_MACCORMACK_FLUIDNET) {
    // We're done. The forward Euler step is already in the output array.
    //
    //
    //   ADD OUTFLOW HERE
    //std::cout << "WE ARE JUST PERFORMING EULER " << std::endl;
    if (Openbound){
        T vel = ApplyOutflow(flags,U,orig,dt,1);
        U_dst = vel;
    }

  } else {
    // Otherwise we need to do the backwards step (which is a SemiLagrange
    // step on the forward data - hence we needed to finish the above loops
    // before moving on).
    bwd.select(1,0).masked_scatter_(maskBorder.squeeze(1),
         pos_corrected.select(1,0).masked_select(maskBorder.squeeze(1)));
    bwd.select(1,1).masked_scatter_(maskBorder.squeeze(1),
         pos_corrected.select(1,1).masked_select(maskBorder.squeeze(1)));
    if (is3D) {
      bwd.select(1,2).masked_scatter_(maskBorder.squeeze(1),
           pos_corrected.select(1,2).masked_select(maskBorder.squeeze(1)));
    }

    // Backward step.
    if (method == ADVECT_MACCORMACK_FLUIDNET) {
      bwd.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
            SemiLagrangeEulerFluidNetMAC(flags, U, fwd, maskBorder, -dt,
            order_space, line_trace, idx_x, idx_y, idx_z)
            .select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
      bwd.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
            SemiLagrangeEulerFluidNetMAC(flags, U, fwd, maskBorder, -dt,
            order_space, line_trace, idx_x, idx_y, idx_z)
            .select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
      if (is3D) {
        bwd.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
              SemiLagrangeEulerFluidNetMAC(flags, U, fwd, maskBorder, -dt,
              order_space, line_trace, idx_x, idx_y, idx_z)
              .select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
      }
    }

    // Now compute the correction.
    T CorrectMAC = MacCormackCorrectMAC(flags, orig, fwd, bwd,
                                       maccormack_strength, idx_x, idx_y, idx_z);
    U_dst.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
          CorrectMAC.select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
    U_dst.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
          CorrectMAC.select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
    if (is3D) {
      U_dst.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
            CorrectMAC.select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
    }

    // Now perform clamping.
    const T dval = at::zeros({bsz, 3, d, h, w}, options).toType(U.scalar_type());
    dval.select(1,0) = U_dst.select(1,0).clone();
    dval.select(1,1) = U_dst.select(1,1).clone();
    if (is3D) {
      dval.select(1,2) = U_dst.select(1,2).clone();
    }

    T ClampMAC = MacCormackClampMAC(flags, U, dval, orig, fwd, maskBorder,
                                    dt, idx_x, idx_y, idx_z);
    U_dst.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
             ClampMAC.select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
    U_dst.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
             ClampMAC.select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
    if (is3D) {
      U_dst.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
               ClampMAC.select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
    }

    if (Openbound){
        T vel = ApplyOutflow(flags,U,orig,dt,1);
        U_dst = vel;
    }

  }
  return U_dst;
}

std::vector<T> solveLinearSystemJacobi
(
   T flags,
   T div,
   const bool is3D,
   const float p_tol = 1e-5,
   const int max_iter = 1000,
   const bool verbose = false
) {
  auto options = div.options();

  // Check arguments.
  T p = zeros_like(flags);
  AT_ASSERTM(p.dim() == 5 && flags.dim() == 5 && div.dim() == 5,
             "Dimension mismatch");
  AT_ASSERTM(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  int numel = d * h * w;
  AT_ASSERTM(p.is_same_size(flags), "size mismatch");
  AT_ASSERTM(div.is_same_size(flags), "size mismatch");
  if (!is3D) {
    AT_ASSERTM(d == 1, "d > 1 for a 2D domain");
  }

  AT_ASSERTM(p.is_contiguous() && flags.is_contiguous() &&
            div.is_contiguous(), "Input is not contiguous");

  T p_prev = at::zeros({bsz, 1, d, h, w}, options).toType(p.scalar_type());
  T p_delta = at::zeros({bsz, 1, d, h, w}, options).toType(p.scalar_type());
  T p_delta_norm = at::zeros({bsz}, options).toType(p.scalar_type());

  if (max_iter < 1) {
     AT_ERROR("At least 1 iteration is needed (maxIter < 1)");
  }

  // Initialize the pressure to zero.
  p.zero_();

  // Start with the output of the next iteration going to pressure.
  T* cur_p = &p;
  T* cur_p_prev = &p_prev;
  //RealGrid* cur_pressure_prev = &pressure_prev;

  T residual;


  int64_t iter = 0;
  while (true) {
    const int32_t bnd =1;
    // Kernel: Jacobi Iteration
    T mCont = at::ones({bsz, 1, d, h, w}, options).toType(at::kByte); // Continue mask

    T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
    T idx_y = at::arange(0, h, options).view({1,h,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
    T idx_z = zeros_like(idx_x);

    if (is3D) {
       idx_z = at::arange(0, d, options).view({1,d,1,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
    }

    T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(at::kLong);
    idx_b = idx_b.expand({bsz,d,h,w});


    T maskBorder = (idx_x < bnd).__or__
                   (idx_x > w - 1 - bnd).__or__
                   (idx_y < bnd).__or__
                   (idx_y > h - 1 - bnd);
    if (is3D) {
        maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                      (idx_z > d - 1 - bnd);
    }
    maskBorder.unsqueeze_(1);
    //T maskBorder = flags.eq(TypeObstacle);

    cur_p->masked_fill_(maskBorder, 0);
    mCont.masked_fill_(maskBorder, 0);


    T maskObstacle = flags.eq(TypeObstacle).__and__(mCont.eq(1));
    cur_p->masked_fill_(maskObstacle, 0);
    mCont.masked_fill_(maskObstacle, 0);

    T zero_f = at::zeros_like(p); // Floating zero
    T zero_l = at::zeros_like(p).toType(at::kLong); // Long zero (for index)
    T zeroBy = at::zeros_like(p).toType(at::kByte); // Long zero (for index)
    // Otherwise, we are in a fluid or empty cell.
    // First, we get all the neighbors.

    T pC = *cur_p_prev;


    T i_l = zero_l.where( (idx_x <=0), idx_x - 1);
    T p1 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_l})
        .unsqueeze(1));

    T i_r = zero_l.where( (idx_x > w - 1 - bnd), idx_x + 1);
    T p2 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_r})
        .unsqueeze(1));

    T j_l = zero_l.where( (idx_y <= 0), idx_y - 1);
    T p3 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_l, idx_x})
        .unsqueeze(1));
    T j_r = zero_l.where( (idx_y > h - 1 - bnd), idx_y + 1);
    T p4 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_r, idx_x})
        .unsqueeze(1));

    T k_l = zero_l.where( (idx_z <= 0), idx_z - 1);
    T p5 = is3D ? zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_l, idx_y, idx_x})
        .unsqueeze(1)) : zero_f;
    T k_r = zero_l.where( (idx_z > d - 1 - bnd), idx_z + 1);
    T p6 = is3D ? zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_r, idx_y, idx_x})
        .unsqueeze(1)) : zero_f;


    T neighborLeftObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeObstacle).toType(at::kByte)).unsqueeze(1));
    T neighborRightObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeObstacle).toType(at::kByte)).unsqueeze(1));
    T neighborBotObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeObstacle).toType(at::kByte)).unsqueeze(1));
    T neighborUpObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeObstacle).toType(at::kByte)).unsqueeze(1));


    T neighborBackObs = zeroBy;
    T neighborFrontObs = zeroBy;


    if (is3D) {
      T neighborBackObs = mCont.__and__(zeroBy.
           where(mCont.ne(1), flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeObstacle).toType(at::kByte)).unsqueeze(1));
      T neighborFrontObs = mCont.__and__(zeroBy.
           where(mCont.ne(1), flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeObstacle).toType(at::kByte)).unsqueeze(1));
    }

    p1.masked_scatter_(neighborLeftObs.eq(1), pC.masked_select(neighborLeftObs.eq(1)));
    p2.masked_scatter_(neighborRightObs.eq(1), pC.masked_select(neighborRightObs.eq(1)));
    p3.masked_scatter_(neighborBotObs.eq(1), pC.masked_select(neighborBotObs.eq(1)));
    p4.masked_scatter_(neighborUpObs.eq(1), pC.masked_select(neighborUpObs.eq(1)));
    p5.masked_scatter_(neighborBackObs.eq(1), pC.masked_select(neighborBackObs.eq(1)));
    p6.masked_scatter_(neighborFrontObs.eq(1), pC.masked_select(neighborFrontObs.eq(1)));



    const float denom = is3D ? 6 : 4;
    (*cur_p).masked_scatter_(mCont.eq(1), (
                (p1 + p2 + p3 + p4 + p5 + p6 + div) / (denom)).masked_select(mCont.eq(1)));


    // Currrent iteration output is now in cur_pressure

    // Now calculate the change in pressure up to a sign (the sign might be
    // incorrect, but we don't care).
    // p_delta = p - p_prev
    at::sub_out(p_delta, p, p_prev);
    p_delta.resize_({bsz, numel});

    // Calculate L2 norm over dim 2.
    at::norm_out(p_delta_norm, p_delta, at::Scalar(2), 1);
    p_delta.resize_({bsz, 1, d, h, w});
    residual = p_delta_norm.max();
    if (verbose) {
      //std::cout << "Jacobi iteration " << (iter + 1) << ": residual "
      //          << residual << std::endl;
    }


    if (residual.item().to<float>() < p_tol) {
      if (verbose) {
        std::cout << "Jacobi max residual fell below p_tol (" << p_tol
                  << ") (terminating)" << std::endl;
        std::cout << "Residual: --------------------------------------> " << residual.item().to<float>()  << std::endl;
      }
      break;
    }

    iter++;
    if (iter >= max_iter) {
        if (verbose) {
          std::cout << "Jacobi max iteration count (" << max_iter
                    << ") reached (terminating)" << std::endl;
          std::cout << "Residual: --------------------------------------> " << residual.item().to<float>()  << std::endl;
        }
        break;
    }

    // We haven't yet terminated.
    auto tmp = cur_p;
    cur_p = cur_p_prev;
    cur_p_prev = tmp;
  } // end while

  // If we terminated with the cur_pressure pointing to the tmp array, then we
  // have to copy the pressure back into the output tensor.
  if (cur_p == &p_prev) {
    p.copy_(p_prev);  // p = p_prev
  }

  // TODO: write mean-subtraction (FluidNet does it in Lua)
  return {p, residual};
}


// We declare the Preconditioner module

T Precon_Z
(
    T& flags,
    T& div,
    T& residual,
    T& Precon,
    T& A_next_i,
    T& A_next_j
){

    //const int batch = 1;

    int batch = flags.size(0)-1;
    //int d = flags.size(2);
    int h = flags.size(3);
    int w = flags.size(4);
    //bool is3D = (d > 1);

    T Temporal_1 = zeros_like(div); // Floating zero
    T Temporal_2 = zeros_like(div); // Floating zero
    T Q = zeros_like(div); // Floating zero
    T Z = zeros_like(div); // Floating zero


    // for loop execution. Now we are in a 2D case ... we'll see later on the 3D implementation

    //std::cout << "Debug 12 Z" << std::endl;

    for( int i = 0; i < w; i = i + 1 ) {
        for( int j = 0; j < h; j = j + 1 ) {


            T Intermediate = flags[batch][0][0][j][i];

            float flag_val = Intermediate.item().to<float>();
            bool isfluid = (flag_val < 2.0);

            //If the corresponding position is fluid: Algo from Bridson pg 65
            if (isfluid) {
               Z[batch][0][0][j][i] = Precon[batch][0][0][j][i] * (
                                      residual[batch][0][0][j][i]
                                      - (A_next_i[batch][0][0][j][i-1]*Precon[batch][0][0][j][i-1]*Z[batch][0][0][j][i-1])
                                      - (A_next_j[batch][0][0][j-1][i]*Precon[batch][0][0][j-1][i]*Z[batch][0][0][j-1][i])  );

            }

        }
    }

    // Now we attack the equation L^T z = q
    // for loop execution. Now we are in a 2D case ... we'll see later on the 3D implementation

    for( int i = w-1; i > 0; i = i - 1 ) {
        for( int j = h-1; j > 0; j = j - 1 ) {

            T Intermediate = flags[batch][0][0][j][i];
            float flag_val = Intermediate.item().to<float>();

            bool isfluid = (flag_val < 2.0);

            //If the corresponding position is fluid: Algo from Bridson pg 65
            if (isfluid) {
                Z[batch][0][0][j][i] = Precon[batch][0][0][j][i]* (
                                           Z[batch][0][0][j][i]
                                           - (A_next_i[batch][0][0][j][i]*Precon[batch][0][0][j][i]*Z[batch][0][0][j][i+1])
                                           - (A_next_j[batch][0][0][j][i]*Precon[batch][0][0][j][i]*Z[batch][0][0][j+1][i])   );


            }
        }
    }

    return Z;
}


// We now declare the PCG solving

std::vector<T> solveLinearSystemPCG
(
     T flags,
     T div,
     T inflow,
     const bool is3D,
     const float p_tol = 1e-5,
     const int max_iter = 1000,
     const bool verbose = false
) {
        auto t_init_i = std::chrono::high_resolution_clock::now();
        auto options = div.options();

        // Check arguments.
        T p = zeros_like(flags);
        T mean_grid = ones_like(flags);

	    // Start with the output of the next iteration going to pressure.
	    T cur_p = p;


        // Create the A_diagonal and A_next_i,  A_next_j, A_next_k

        T A_diag = zeros_like(flags);

        T A_left = ones_like(flags);
        T A_bot = ones_like(flags);
        T A_back = ones_like(flags);
        T A_right = ones_like(flags);
        T A_up = ones_like(flags);
        T A_front = ones_like(flags);
        T A_inflow = zeros_like(flags);
        T A_fluid = zeros_like(flags);
        T A_next_i = zeros_like(flags);
        T A_next_j = zeros_like(flags);
        T A_next_k = zeros_like(flags);



        // Constant initializing

        const int32_t bnd =1;
        const float dt = 5;

        AT_ASSERTM(p.dim() == 5 && flags.dim() == 5 && div.dim() == 5,
                   "Dimension mismatch");
        AT_ASSERTM(flags.size(1) == 1, "flags is not scalar");
        int bsz = flags.size(0);
        int d = flags.size(2);
        int h = flags.size(3);
        int w = flags.size(4);

        //int numel = d * h * w;
        AT_ASSERTM(p.is_same_size(flags), "size mismatch");
        AT_ASSERTM(div.is_same_size(flags), "size mismatch");
        if (!is3D) {
            AT_ASSERTM(d == 1, "d > 1 for a 2D domain");
        }

        AT_ASSERTM(p.is_contiguous() && flags.is_contiguous() &&
                   div.is_contiguous(), "Input is not contiguous");

        // Kernel: Jacobi Iteration
        T mCont = at::ones({bsz, 1, d, h, w}, options).toType(at::kByte); // Continue mask

        T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
        T idx_y = at::arange(0, h, options).view({1,h,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
        T idx_z = zeros_like(idx_x);
        if (is3D) {
            idx_z = at::arange(0, d, options).view({1,d,1,1}).expand({bsz, d, h, w}).toType(idx_x.scalar_type());
        }

        T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(at::kLong);
        idx_b = idx_b.expand({bsz,d,h,w});

        T maskBorder = (idx_x < bnd).__or__
        (idx_x > w - 1 - bnd).__or__
        (idx_y < bnd).__or__
        (idx_y > h - 1 - bnd);
        if (is3D) {
            maskBorder = maskBorder.__or__(idx_z < bnd).__or__
            (idx_z > d - 1 - bnd);
        }
        maskBorder.unsqueeze_(1);

        // Zero pressure on the border.
        cur_p.masked_fill_(maskBorder, 0);
        mCont.masked_fill_(maskBorder, 0);

        T zero_f = at::zeros_like(p); // Floating zero
        T zero_l = at::zeros_like(p).toType(at::kLong); // Long zero (for index)
        T zeroBy = at::zeros_like(p).toType(at::kByte); // Long zero (for index)
        T oneBy = at::ones_like(p).toType(at::kByte); // Long zero (for index)
        // Otherwise, we are in a fluid or empty cell.
        // First, we get all the neighbors.

        //T pC = *cur_p_prev;

        T i_l = zero_l.where( (idx_x <=0), idx_x - 1);
        //T p1 = zero_f.
        //    where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_l})
        //    .unsqueeze(1));

        T i_r = zero_l.where( (idx_x > w - 1 - bnd), idx_x + 1);
        //T p2 = zero_f.
        //    where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_r})
        //    .unsqueeze(1));

        T j_l = zero_l.where( (idx_y <= 0), idx_y - 1);
        //T p3 = zero_f.
        //    where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_l, idx_x})
        //    .unsqueeze(1));
        T j_r = zero_l.where( (idx_y > h - 1 - bnd), idx_y + 1);
        //T p4 = zero_f.
        //    where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_r, idx_x})
        //    .unsqueeze(1));
        T k_l = zero_l.where( (idx_z <= 0), idx_z - 1);
        //T p5 = is3D ? zero_f.
        //    where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_l, idx_y, idx_x})
        //    .unsqueeze(1)) : zero_f;
        T k_r = zero_l.where( (idx_z > d - 1 - bnd), idx_z + 1);
        //T p6 = is3D ? zero_f.
        //    where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_r, idx_y, idx_x})
        //    .unsqueeze(1)) : zero_f;

        T neighborLeftObs = mCont.__and__(zeroBy.
                                          where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeObstacle)));
        T neighborRightObs = mCont.__and__(zeroBy.
                                           where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeObstacle)));
        T neighborBotObs = mCont.__and__(zeroBy.
                                            where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeObstacle)));
        T neighborUpObs = mCont.__and__(zeroBy.
                                        where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeObstacle)));
        T neighborBackObs = oneBy;
        T neighborFrontObs = oneBy;

        if (is3D) {
            T neighborBackObs = mCont.__and__(zeroBy.
                                              where(mCont.ne(1), flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeObstacle)));
            T neighborFrontObs = mCont.__and__(zeroBy.
                                               where(mCont.ne(1), flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeObstacle)));
        }


        // Depending on the dimension our fluid will start on 2 or 3 (Laplacian scheme)
        const float dnom = is3D ? 3 : 2;

        //The basic flag where 1 = fluid (objects and borders = 0)
        //T fluid_flag = mCont.__and__(zeroBy.
        //                             where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeObstacle)).unsqueeze(1));


        //The basic flag where 1 = fluid (objects and borders = 0)
        T fluid_flag = flags.index({idx_b, zero_l, idx_z, idx_y, idx_x}).eq(TypeFluid);

        // 1 where fluid and no borcer in the left/bottom/back
        A_left.masked_fill_(neighborLeftObs, 0);
        A_bot.masked_fill_(neighborBotObs, 0);
        A_back.masked_fill_(neighborBackObs, 0);

        A_right.masked_fill_(neighborRightObs, 0);
        A_up.masked_fill_(neighborUpObs, 0);
        A_front.masked_fill_(neighborFrontObs, 0);

        A_left.masked_fill_(fluid_flag.ne(1), 0);
        A_bot.masked_fill_(fluid_flag.ne(1), 0);
        A_back.masked_fill_(fluid_flag.ne(1), 0);

        A_right.masked_fill_(fluid_flag.ne(1), 0);
        A_up.masked_fill_(fluid_flag.ne(1), 0);
        A_front.masked_fill_(fluid_flag.ne(1), 0);


        A_inflow.masked_fill_(inflow>0.01, 1);
        A_fluid.masked_fill_(fluid_flag, dnom);


        //Watch the A_diagonal structure
        //A_diag = A_fluid+A_left+A_bot+A_back;
        A_diag = A_left+A_bot+A_back+A_right+A_up+A_front+A_inflow;
        //A_diag = A_diag * dt;

        //Similar procedure for the A_next_i/j/k
        //We will have -1 when the cell is fluid and the cellule to the right/top/front is fluid
        A_next_i.masked_fill_(neighborRightObs.ne(1),-1);
        A_next_j.masked_fill_(neighborUpObs.ne(1),-1);
        A_next_k.masked_fill_(neighborFrontObs.ne(1),-1);


        A_next_i.masked_fill_(mCont.ne(1),0);
        A_next_j.masked_fill_(mCont.ne(1),0);
        A_next_k.masked_fill_(mCont.ne(1),0);


        mean_grid.masked_fill_(mCont.ne(1),0);
        T mean_grid_mono = mean_grid.sum();


        if (max_iter < 1) {
            AT_ERROR("At least 1 iteration is needed (maxIter < 1)");
        }


        // Initialization
        // We will create a diagonal mask + a mask for the terms outside the diagonal
        //

        T ones = ones_like(flags).toType(at::kByte);;
        T zeros = zeros_like(flags).toType(at::kByte);;

        T Diag_mask = fluid_flag.__and__(where(idx_x.eq(idx_y),ones,zeros));
        T I_plus_mask = fluid_flag.__and__(where(i_r.eq(idx_y),ones,zeros));
        T I_minus_mask = fluid_flag.__and__(where(i_l.eq(idx_y),ones,zeros));
        T J_plus_mask = fluid_flag.__and__(where(idx_x.eq(j_r),ones,zeros));
        T J_minus_mask = fluid_flag.__and__(where(idx_x.eq(j_l),ones,zeros));

        p.zero_();


        //
        // STEP 2
        //
        // Set the residual = b (div)
        //

        T residual = div.clone();

       // We will first calculate the preconditioner

        // We declare the tuning constant (tau) and the safety constant (sigma)
        // E_diag and Precon will have the size i,j, k (same as flags)
        const float tau = 0.97;
        const float Safety = 0.25;
        //const int k = 0;

        T E_diag = zeros_like(div); // Floating zero
        T Precon = zeros_like(div); // Floating zero
        T W = zeros_like(div); // Floating zero
        T z = zeros_like(div); // Floating zero

        // BATCH LOOP!!!!!!
        //
        //

        bool mask[h][w];
        bool isSmall;

        auto t_init_b_i = std::chrono::high_resolution_clock::now();

        for( int batch=0; batch<bsz; batch = batch +1){

            // for loop execution. Now we are in a 2D case ... we'll see later on the 3D implementation
            //bool mask[w][h];
            for( int i = 0; i < w; i = i + 1 ) {
                for( int j = 0; j < h; j = j + 1 ) {
                    T Intermediate = flags[batch][0][0][j][i];
                    float flag_val = Intermediate.item().to<float>();
                    mask[j][i] = (flag_val < 2.0);
                }
            }

        }
        auto t_init_b_inter = std::chrono::high_resolution_clock::now();

        for( int batch=0; batch<bsz; batch = batch +1){

            std::cout << "BATCH ------------------        :"<< batch << std::endl;

            for( int i = 0; i < w; i = i + 1 ) {
                for( int j = 0; j < h; j = j + 1 ) {

                    if (mask[j][i]) {
                       auto t_inside_1 = std::chrono::high_resolution_clock::now();

                       E_diag[batch][0][0][j][i] = A_diag[batch][0][0][j][i]
                                                     -  ((A_next_i[batch][0][0][j][i-1]*Precon[batch][0][0][j][i-1])*
                                                             (A_next_i[batch][0][0][j][i-1]*Precon[batch][0][0][j][i-1]))
                                                     -  ((A_next_j[batch][0][0][j-1][i]*Precon[batch][0][0][j-1][i])*
                                                             (A_next_j[batch][0][0][j-1][i]*Precon[batch][0][0][j-1][i]))
                                                     -  tau * (
                                                            A_next_i[batch][0][0][j][i-1]*A_next_j[batch][0][0][j][i-1]*
                                                            ((Precon[batch][0][0][j][i-1])*(Precon[batch][0][0][j][i-1]))  +
                                                            A_next_j[batch][0][0][j-1][i]*A_next_i[batch][0][0][j-1][i]*
                                                            ((Precon[batch][0][0][j-1][i])*(Precon[batch][0][0][j-1][i]))   );

                       auto t_inside_2 = std::chrono::high_resolution_clock::now();


                       auto t_inside_25 = std::chrono::high_resolution_clock::now();


                       auto t_inside_3 = std::chrono::high_resolution_clock::now();


                       T Inter_1 = ( E_diag[batch][0][0][j][i] - Safety * A_diag[batch][0][0][j][i]);
                       float Diag_value = Inter_1.item().to<float>();
                       bool isSmall = (Diag_value < 0);

                       if (isSmall) {
                           E_diag[batch][0][0][j][i] = A_diag[batch][0][0][j][i];
                        }

                       auto t_inside_4 = std::chrono::high_resolution_clock::now();

                       //Finally, the precondtionner will be equal to 1/sqrt(E)
                       Precon[batch][0][0][j][i]= 1 / ((E_diag[batch][0][0][j][i]).pow(0.5));

                       auto t_inside_5 = std::chrono::high_resolution_clock::now();

                       auto duration_inside_1 = std::chrono::duration_cast<std::chrono::microseconds>( t_inside_2 - t_inside_1 ).count();
                       auto duration_inside_2 = std::chrono::duration_cast<std::chrono::microseconds>( t_inside_25 - t_inside_2 ).count();
                       auto duration_inside_25 = std::chrono::duration_cast<std::chrono::microseconds>( t_inside_3 - t_inside_25 ).count();
                       auto duration_inside_3 = std::chrono::duration_cast<std::chrono::microseconds>( t_inside_4 - t_inside_3 ).count();
                       auto duration_inside_4 = std::chrono::duration_cast<std::chrono::microseconds>( t_inside_5 - t_inside_4 ).count();

                       std::cout << "Duration inside  Boucle 1 -------- "<< duration_inside_1 << std::endl;
                       std::cout << "Duration inside  Boucle 2 -------- "<< duration_inside_2 << std::endl;
                       std::cout << "Duration inside  Boucle 25 -------- "<< duration_inside_25 << std::endl;
                       std::cout << "Duration inside  Boucle 3 -------- "<< duration_inside_3 << std::endl;
                       std::cout << "Duration inside  Boucle 4 -------- "<< duration_inside_4 << std::endl;

                   }
                }
            }
            auto t_init_b_f = std::chrono::high_resolution_clock::now();


            // Once we have calculated the preconditioner, we should now solve z = Mr
            //T z = Precon_Z(flags, div, residual, Precon, A_next_i,A_next_j);
            T z = residual.clone();
            // At this point we will get our first sigma:
            // Sigma = z*r
            // We first calculate the mono dim Tensor, then cast it to a pointer and then to a float
            //
            // 100% Sure there is a WAYYYY better method

            //
            //After applying the preconditioner to  the residual to get z,
            //We change the search tensor s=z

            T s = z.clone();
            //T s = residual.clone();


            T Sigma_mono = (residual * residual).sum();
            float Sigma = Sigma_mono.item().to<float>();

            // We now initialize the search and residual_max tensor and the alpha/beta ctes.

            T residual_max = zeros_like(Sigma_mono);

            float beta=0;

            //Declare the Neighbours for the operation z = A * s
            //There should be a better method I suppose

            T S_diag = zeros_like(div);
            T S_minus_i = zeros_like(div);
            T S_plus_i = zeros_like(div);
            T S_minus_j = zeros_like(div);
            T S_plus_j = zeros_like(div);

            T P_minus_i = zeros_like(div);
            T P_plus_i = zeros_like(div);
            T P_minus_j = zeros_like(div);
            T P_plus_j = zeros_like(div);

            T W = zeros_like(div);

            S_diag = s.clone();
            S_minus_i = s.clone();
            S_plus_i = s.clone();
            S_minus_j = s.clone();
            S_plus_j = s.clone();

            S_diag.masked_fill_(Diag_mask.ne(1),0);
            S_minus_i.masked_fill_(I_minus_mask.ne(1),0);
            S_plus_i.masked_fill_(I_plus_mask.ne(1),0);
            S_minus_j.masked_fill_(J_minus_mask.ne(1),0);
            S_plus_j.masked_fill_(J_plus_mask.ne(1),0);

            // We initialize the iterations and we start the cycle.
            // It will run until the residual  is too small
            int64_t iter = 0;

            auto t_init_f = std::chrono::high_resolution_clock::now();
            auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>( t_init_f - t_init_i ).count();
            auto duration_init_inter = std::chrono::duration_cast<std::chrono::microseconds>( t_init_b_inter - t_init_b_i ).count();
            auto duration_init_b = std::chrono::duration_cast<std::chrono::microseconds>( t_init_b_f - t_init_b_i ).count();
            //std::cout << "Duration 3 "<< duration_3 << std::endl
            std::cout << "Duration INIT  -------- "<< duration_init << std::endl;
            std::cout << "Duration INIT  Boucle -------- "<< duration_init_b << std::endl;
            std::cout << "Duration INIT  Mask Boucle -------- "<< duration_init_inter << std::endl;
            //Debug Printing
            //
            //std::cout << "Debug 15" << std::endl;

            while (true) {


                residual_max = (abs(residual)).max();

                auto t0 = std::chrono::high_resolution_clock::now();

                for( int i = 0; i < w; i = i + 1 ) {
                   for( int j = 0; j < h; j = j + 1 ) {

                       if(mask[j][i]){

                           //auto t3 = std::chrono::high_resolution_clock::now();
                           W[batch][0][0][j][i] = (A_diag[batch][0][0][j][i]*s[batch][0][0][j][i]) +
                                    (A_next_i[batch][0][0][j][i-1]*s[batch][0][0][j][i-1]) +(A_next_i[batch][0][0][j][i]*s[batch][0][0][j][i+1])+
                                    (A_next_j[batch][0][0][j-1][i]*s[batch][0][0][j-1][i]) + (A_next_j[batch][0][0][j][i]*s[batch][0][0][j+1][i]);

                        }
                    }
                }
                auto t5 = std::chrono::high_resolution_clock::now();
                auto duration_4 = std::chrono::duration_cast<std::chrono::microseconds>( t5 - t0 ).count();


                T Inter_Sigma_mono = (s*W).sum();
                float Inter_Sigma = Inter_Sigma_mono.item().to<float>();

                std::cout << "Sigma " << Sigma  << std::endl;
                std::cout << "Inter Sigma " << Inter_Sigma  << std::endl;

                float alpha=0;

                if (abs(Inter_Sigma) > 0.0001) {
                   alpha = Sigma / Inter_Sigma;
                }


                p = p + alpha*s;

                T mean_p_mono = p.sum();
                float mean_p = (mean_p_mono/mean_grid_mono).item().to<float>();
                p = p - mean_p;
                p.masked_fill_(mCont.ne(1),0);

                residual = residual - alpha*W;


                std::cout << "Residual Max: ------------------------> " << (abs(residual).max()).item().to<float>()  << std::endl;
                std::cout << "Residual 2 Sum: ----------------------> " << ((residual*residual).sum()).item().to<float>()  << std::endl;

                residual_max = (abs(residual)).max();

                if (residual_max.item().to<float>() < p_tol) {
                   if (verbose) {
                          std::cout << "The residual is already small enough, P = 0" << std::endl;
                           }
                    break;
                }


                std::cout << "Iter: " << iter << " Max Tol "<<  p_tol << std::endl;

                if (verbose) {
                    std::cout << "PCG iteration " << (iter + 1) << ": residual "
                    << residual << std::endl;
                }


                // We redo a new sigma = z*r
                T New_Sigma_mono = (residual*residual).sum();
                float New_Sigma = New_Sigma_mono.item().to<float>();

                float beta=0;

                if (abs(alpha) > 0.0001) {
                   beta = New_Sigma / Sigma;
                }
                beta = New_Sigma / Sigma;

                std::cout << "beta Sigma " << Sigma  << std::endl;
                std::cout << "beta New_Sigma " << New_Sigma  << std::endl;
                std::cout << "BETA                            " << beta  << std::endl;

                s = residual + beta*s;
                Sigma = New_Sigma;

                iter++;
                if (iter >= max_iter) {
                    if (verbose) {
                        std::cout << "PCG  max iteration count (" << max_iter
                        << ") reached (terminating)" << std::endl;
                    }
                    break;
                }
            } // end while
            return {p, residual};

    } // end boucle batch

}//end declaration PCG

void solveLinearSystemCG
(
   T flags,
   T div_vec,
   T A_val,
   T I_A,
   T J_A,
   T p,
   float &residue,
   const bool is3D,
   const float p_tol = 1e-5,
   const int max_iter = 1000,
   const bool verbose = false
) {

  // Check arguments.
  AT_ASSERTM(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  int numel = d * h * w;

  int nnz = A_val.size(0);

  Conjugate_Gradient(flags,div_vec,A_val, I_A, J_A, p_tol, max_iter, h*w*d,h*w*d,nnz, p, residue);


} // End CG

} // namespace fluid



template<class InputIt1, class InputIt2, class T>
T inner_product(InputIt1 first1, InputIt1 last1,
                    InputIt2 first2, T init)
{
        while (first1 != last1) {
            init = std::move(init) + *first1 * *first2; // std::move since C++20
            ++first1;
            ++first2;
        }
        return init;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("advect_scalar", &(fluid::advectScalar), "Advect Scalar");
    m.def("advect_vel", &(fluid::advectVel), "Advect Velocity");
    m.def("solve_linear_system_Jacobi", &(fluid::solveLinearSystemJacobi), "Solve Linear System using Jacobi's method");
    m.def("solve_linear_system_PCG", &(fluid::solveLinearSystemPCG), "Solve Linear System using PCG method");
    m.def("solve_linear_system_CG", &(fluid::solveLinearSystemCG), "Solve Linear System using CG method");
}
