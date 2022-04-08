#include "grid.h"
#include "cell_type.h"

namespace fluid {

typedef at::Tensor T;

float getDx(at::Tensor self) {
  float gridSizeMax = std::max(std::max(self.size(2), self.size(3)), self.size(4));
  return (1.0 / gridSizeMax);
}

T interpol(const T& self, const T& pos) {

  AT_ASSERTM(pos.size(1) == 3, "Input pos must have 3 channels");

  bool is3D = (self.size(2) > 1);
  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);

  auto options = self.options();

  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T pos0 = p.toType(at::kLong);

  T s1 = (p.select(1,0) - pos0.select(1,0).toType(self.scalar_type())).unsqueeze(1);
  T t1 = (p.select(1,1) - pos0.select(1,1).toType(self.scalar_type())).unsqueeze(1);
  T f1 = (p.select(1,2) - pos0.select(1,2).toType(self.scalar_type())).unsqueeze(1);
  T s0 = 1 - s1;
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  T x0 = pos0.select(1,0).clamp_(0, self.size(4) - 2);
  T y0 = pos0.select(1,1).clamp_(0, self.size(3) - 2);
  T z0 = pos0.select(1,2).clamp_(0, self.size(2) - 2);

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(x0.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);

  // Alguacil: when performing index() in torch, if dimensions are not contiguous,
  // it transposes everything to the front. We transpose it back to the chan dimension 1.
  if (is3D) {
   T Ia= self.index({idx_b, {}, z0  , y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib= self.index({idx_b, {}, z0  , y0+1, x0  }).squeeze(4).unsqueeze(1);
   T Ic= self.index({idx_b, {}, z0  , y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id= self.index({idx_b, {}, z0  , y0+1, x0+1}).squeeze(4).unsqueeze(1);
   T Ie= self.index({idx_b, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1);
   T If= self.index({idx_b, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1);
   T Ig= self.index({idx_b, {}, z0+1, y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Ih= self.index({idx_b, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1);

    return ( ((Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1)*f0 +
             ((Ie*t0 + If*t1)*s0 + (Ig*t0 + Ih*t1)*s1)*f1 );
  } else {
   T Ia= self.index({idx_b, {}, z0, y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib= self.index({idx_b, {}, z0, y0+1, x0  }).squeeze(4).unsqueeze(1);
   T Ic= self.index({idx_b, {}, z0, y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id= self.index({idx_b, {}, z0, y0+1, x0+1}).squeeze(4).unsqueeze(1);

    return ( (Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1 );
  }
}

void interpol1DWithFluid(
    const T& val_a, const T& is_fluid_a,
    const T& val_b, const T& is_fluid_b,
    const T& t_a, const T& t_b,
    T& is_fluid_ab, T& val_ab) {

  T m0 = is_fluid_a.eq(0).__and__(is_fluid_b.eq(0));
  T m1 = is_fluid_a.eq(0).__and__(m0.eq(0));
  T m2 = is_fluid_b.eq(0).__and__(m1.eq(0)).__and__(m0.eq(0));

  //T m3 = 1 - (m0.__or__(m1).__or__(m2));
  T m3_inter = (m0.__or__(m1).__or__(m2));
  T m3 = m3_inter.eq(0);


  val_ab = zeros_like(val_a);
  val_ab.masked_fill_(m0, 0);
  val_ab.masked_scatter_(m1, val_b.masked_select(m1));
  val_ab.masked_scatter_(m2, val_a.masked_select(m2));
  val_ab.masked_scatter_(m3, (val_a*t_a + val_b*t_b).masked_select(m3));


  is_fluid_ab = m0.eq(0);
}

void interpol1DWithFluidTest(
    const T& val_a, const T& is_fluid_a,
    const T& val_b, const T& is_fluid_b,
    const T& t_a, const T& t_b,
    T& is_fluid_ab, T& val_ab) {

  T m0 = is_fluid_a.eq(0).__and__(is_fluid_b.eq(0));
  T m1 = is_fluid_a.eq(0).__and__(m0.eq(0));
  T m2 = is_fluid_b.eq(0).__and__(m1.eq(0)).__and__(m0.eq(0));
  T m3 = 1 - (m0.__or__(m1).__or__(m2));

  val_ab = zeros_like(val_a);
  val_ab.masked_fill_(m0, 0);
  val_ab.masked_scatter_(m1, val_b.masked_select(m1));
  val_ab.masked_scatter_(m2, val_a.masked_select(m2));
  val_ab.masked_scatter_(m3, (val_a*t_a + val_b*t_b).masked_select(m3));

  is_fluid_ab = m0.eq(0);
}

T interpolWithFluid(const T& self, const T& flags, const T& pos) {

  AT_ASSERTM(pos.size(1) == 3, "Input pos must have 3 channels");

  bool is3D = (self.size(2) > 1);
  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);

  auto options = self.options();

  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T pos0 = p.toType(at::kLong);

  T s1 = (p.select(1,0) - pos0.select(1,0).toType(self.scalar_type())).unsqueeze(1);
  T t1 = (p.select(1,1) - pos0.select(1,1).toType(self.scalar_type())).unsqueeze(1);
  T f1 = (p.select(1,2) - pos0.select(1,2).toType(self.scalar_type())).unsqueeze(1);
  T s0 = 1 - s1;
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  T x0 = pos0.select(1,0).clamp_(0, self.size(4) - 2);
  T y0 = pos0.select(1,1).clamp_(0, self.size(3) - 2);
  T z0 = pos0.select(1,2).clamp_(0, self.size(2) - 2);

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(x0.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);

  if (is3D) {
   // val_ab = data(xi, yi, zi, 0, b) * t0 +
   //          data(xi, yi + 1, zi, 0, b) * t1

   // Alguacil: when performing index() in torch, if dimensions are not contiguous,
   // it transposes everything to the front. We transpose it back to the chan dimension.
   T Ia = self.index({idx_b, {}, z0  , y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib = self.index({idx_b, {}, z0  , y0+1, x0  }).squeeze(4).unsqueeze(1);

   T is_fluid_a = flags.index({idx_b, {}, z0  , y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_b = flags.index({idx_b, {}, z0  , y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Iab = at::empty_like(Ia);
   T is_fluid_ab = at::empty_like(is_fluid_a);
   interpol1DWithFluid(Ia, is_fluid_a, Ib, is_fluid_b, t0, t1, is_fluid_ab, Iab);

   // val_cd = data(xi + 1, yi, zi, 0, b) * t0 +
   //          data(xi + 1, yi + 1, zi, 0, b) * t1
   T Ic = self.index({idx_b, {}, z0  , y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id = self.index({idx_b, {}, z0  , y0+1, x0+1}).squeeze(4).unsqueeze(1);

   T is_fluid_c = flags.index({idx_b, {}, z0  , y0  , x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_d = flags.index({idx_b, {}, z0  , y0+1, x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Icd = at::empty_like(Ic);
   T is_fluid_cd = at::empty_like(is_fluid_c);
   interpol1DWithFluid(Ic, is_fluid_c, Id, is_fluid_d, t0, t1, is_fluid_cd, Icd);

   // val_ef = data(xi, yi, zi + 1, 0, b) * t0 +
   //          data(xi, yi + 1, zi + 1, 0, b) * t1
   T Ie = self.index({idx_b, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1);
   T If = self.index({idx_b, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1);

   T is_fluid_e = flags.index({idx_b, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_f = flags.index({idx_b, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Ief = at::empty_like(Ie);
   T is_fluid_ef = at::empty_like(is_fluid_e);
   interpol1DWithFluid(Ie, is_fluid_e, If, is_fluid_f, t0, t1, is_fluid_ef, Ief);

   // val_gh = data(xi + 1, yi, zi + 1, 0, b) * t0 +
   //          data(xi + 1, yi + 1, zi + 1, 0, b) * t1
   T Ig = self.index({idx_b, {}, z0+1, y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Ih = self.index({idx_b, {}, z0+1, y0+1, x0+1}).squeeze(4).unsqueeze(1);

   T is_fluid_g = flags.index({idx_b, {}, z0+1, y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_h = flags.index({idx_b, {}, z0+1, y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Igh = at::empty_like(Ig);
   T is_fluid_gh = at::empty_like(is_fluid_g);
   interpol1DWithFluid(Ig, is_fluid_g, Ih, is_fluid_h, t0, t1, is_fluid_gh, Igh);

   // val_abcd = val_ab * s0 + val_cd * s1
   T Iabcd = at::empty_like(Igh);
   T is_fluid_abcd = at::empty_like(is_fluid_gh);
   interpol1DWithFluid(Iab, is_fluid_ab, Icd, is_fluid_cd, s0, s1, is_fluid_abcd, Iabcd);

   // val_efgh = val_ef * s0 + val_gh * s1
   T Iefgh = at::empty_like(Iabcd);
   T is_fluid_efgh = at::empty_like(is_fluid_abcd);
   interpol1DWithFluid(Ief, is_fluid_ef, Igh, is_fluid_gh, s0, s1, is_fluid_efgh, Iefgh);

   // val = val_abcd * f0 + val_efgh * f1
   T Ival = at::empty_like(Iabcd);
   T is_fluid = at::empty_like(is_fluid_efgh);
   interpol1DWithFluid(Iabcd, is_fluid_abcd, Iefgh, is_fluid_efgh, f0, f1,
                             is_fluid, Ival);

   T no_fluid = is_fluid.eq(0);
   Ival = Ival.masked_scatter_(no_fluid, interpol(self, pos).masked_select(no_fluid));
   return Ival;

  } else {
   // val_ab = data(xi, yi, 0, 0, b) * t0 +
   //          data(xi, yi + 1, 0, 0, b) * t1

   // Alguacil: when performing index() in torch, if dimensions are not contiguous,
   // it transposes everything to the front. We transpose it back to the chan dimension.

   T Ia = self.index({idx_b, {}, z0 , y0  , x0  }).squeeze(4).unsqueeze(1);
   T Ib = self.index({idx_b, {}, z0 , y0+1, x0  }).squeeze(4).unsqueeze(1);

   T is_fluid_a = flags.index({idx_b, {}, z0 , y0  , x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_b = flags.index({idx_b, {}, z0 , y0+1, x0  }).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Iab = at::empty_like(Ia);
   T is_fluid_ab = at::empty_like(is_fluid_a);
   interpol1DWithFluid(Ia, is_fluid_a, Ib, is_fluid_b, t0, t1, is_fluid_ab, Iab);

   // val_cd = data(xi + 1, yi, 0, 0, b) * t0 +
   //          data(xi + 1, yi + 1, 0, 0, b) * t1
   T Ic = self.index({idx_b, {}, z0 , y0  , x0+1}).squeeze(4).unsqueeze(1);
   T Id = self.index({idx_b, {}, z0 , y0+1, x0+1}).squeeze(4).unsqueeze(1);

   T is_fluid_c = flags.index({idx_b, {}, z0 , y0  , x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);
   T is_fluid_d = flags.index({idx_b, {}, z0 , y0+1, x0+1}).squeeze(4).unsqueeze(1).eq(fluid::TypeFluid);

   T Icd = at::empty_like(Ic);
   T is_fluid_cd = at::empty_like(is_fluid_c);
   interpol1DWithFluid(Ic, is_fluid_c, Id, is_fluid_d, t0, t1, is_fluid_cd, Icd);

   // val = val_ab * s0 + val_cd * s1
   T Ival = at::empty_like(Ic);
   T is_fluid = at::empty_like(is_fluid_cd);
   interpol1DWithFluid(Iab, is_fluid_ab, Icd, is_fluid_cd, s0, s1, is_fluid, Ival);

   T no_fluid = is_fluid.eq(0);
   Ival = Ival.masked_scatter_(no_fluid, interpol(self, pos).masked_select(no_fluid));
   return Ival;
  }
}

// Alguacil: we ensure that this function only call non-edge cells. Index tensors
// are reduced by 2 in all sizes WxHxD. Self is the complete velocity matrix.

T getCentered(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;
  h -= 1;
  w -= 1;

  T idx_x = at::arange(1, w, options).view({1,w-1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_y = at::arange(1, h, options).view({1,h-1, 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d-1, 1 , 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  }

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d-1,h-1,w-1});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T c_vel_x = 0.5 * ( self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x+1}) );
  T c_vel_y = 0.5 * ( self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y+1,idx_x  }) );
  T c_vel_z = at::zeros_like(c_vel_x);

  if (is3D) {
    c_vel_z = 0.5 * ( self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  }) );
  }

  return at::stack({c_vel_x, c_vel_y, c_vel_z}, 1);
}


T getCentered_temp(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;

  T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_y = at::arange(0, h, options).view({1,h, 1}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d, 1 , 1}).expand({bsz,d,h,w}).toType(at::kLong);
  }

  T mns_one = at::ones_like(idx_x); // Floating zero

  T idx_x_new = idx_x.where(idx_x<w-1,(idx_x-(w*mns_one)));
  T idx_y_new = idx_y.where(idx_y<h-1,(idx_y-(h*mns_one)));
  T idx_z_new = idx_z.where(idx_z<d,(idx_z-(d*mns_one)));


  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());


  T c_vel_x = 0.5 * ( self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new  ,idx_x_new  }) +
                    self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new  ,idx_x_new+1}) );
  T c_vel_y = 0.5 * ( self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y_new  ,idx_x_new  }) +
                    self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y_new+1,idx_x_new  }) );
  T c_vel_z = at::zeros_like(c_vel_x);


  if (is3D) {
    c_vel_z = 0.5 * ( self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  }) );
  }

  return at::stack({c_vel_x, c_vel_y, c_vel_z}, 1);
}

// End of getCentered_temp


T getAtMACX(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;
  h -= 1;
  w -= 1;


  T idx_x = at::arange(1, w, options).view({1,w-1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_y = at::arange(1, h, options).view({1,h-1, 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d-1, 1 , 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  }


  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d-1,h-1,w-1});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_x = self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x  });

  T v_y = 0.25 * (self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x-1}) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y+1,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y+1,idx_x-1}) );
  T v_z = at::zeros_like(v_x);

  if (is3D) {
    v_z = 0.25 * (self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x-1}) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x-1}) );
  }
  return at::stack({v_x, v_y, v_z}, 1);


}

// Temporary get MAC
//

T getAtMACX_temp(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;


  T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_y = at::arange(0, h, options).view({1,h, 1}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d, 1 , 1}).expand({bsz,d,h,w}).toType(at::kLong);
  }

  T mns_one = at::ones_like(idx_x); // Floating zero

  T idx_x_new = idx_x.where(idx_x<w-1,(idx_x-(w*mns_one)));
  T idx_y_new = idx_y.where(idx_y<h-1,(idx_y-(h*mns_one)));
  T idx_z_new = idx_z.where(idx_z<d,(idx_z-(d*mns_one)));



  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_x = self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new  ,idx_x_new  });

  T v_y = 0.25 * (self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y_new  ,idx_x_new  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y_new  ,idx_x_new-1}) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y_new+1,idx_x_new  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y_new+1,idx_x_new-1}) );
  T v_z = at::zeros_like(v_x);

  if (is3D) {
    T v_z = 0.25 * (self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x-1}) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x-1}) );
  }

  return at::stack({v_x, v_y, v_z}, 1);
}

T getAtMACY_temp(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;


  T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_y = at::arange(0, h, options).view({1,h, 1}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d, 1 , 1}).expand({bsz,d,h,w}).toType(at::kLong);
  }

  T mns_one = at::ones_like(idx_x); // Floating zero

  T idx_x_new = idx_x.where(idx_x<w-1,(idx_x-(w*mns_one)));
  T idx_y_new = idx_y.where(idx_y<h-1,(idx_y-(h*mns_one)));
  T idx_z_new = idx_z.where(idx_z<d,(idx_z-(d*mns_one)));


  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_x = 0.25 * (self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new  ,idx_x_new  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new-1,idx_x_new  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new  ,idx_x_new+1}) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y_new-1,idx_x_new+1}) );

  T v_y = self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x  });

  T v_z = at::zeros_like(v_x);

  if (is3D) {
    T v_y = 0.25 * (self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y-1,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y-1,idx_x  }) );
  }

  return at::stack({v_x, v_y, v_z}, 1);
}


T getAtMACZ_temp(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;

  T idx_x = at::arange(0, w, options).view({1,w}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_y = at::arange(0, h, options).view({1,h, 1}).expand({bsz,d,h,w}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d, 1 , 1}).expand({bsz,d,h,w}).toType(at::kLong);
  }

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_x = 0.25 * (self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z-1,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x+1}) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z-1,idx_y  ,idx_x+1}) );

  T v_y = 0.25 * (self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z-1,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y+1,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z-1,idx_y+1,idx_x  }) );
  T v_z = at::zeros_like(v_x);

  if (is3D) {
   v_z = self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  });
  }

  return at::stack({v_x, v_y, v_z}, 1);
}


//

T getAtMACY(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;
  h -= 1;
  w -= 1;

  T idx_x = at::arange(1, w, options).view({1,w-1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_y = at::arange(1, h, options).view({1,h-1, 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d-1, 1 , 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  }


  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d-1,h-1,w-1});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_x = 0.25 * (self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y-1,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x+1}) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y-1,idx_x+1}) );

  T v_y = self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x  });

  T v_z = at::zeros_like(v_x);

  if (is3D) {
    v_z = 0.25 * (self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y-1,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y  ,idx_x  }) +
                    self.index({idx_b,idx_c.select(1,2) ,idx_z+1,idx_y-1,idx_x  }) );

  }
  return at::stack({v_x, v_y, v_z}, 1);



}

T getAtMACZ(const T& self) {

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;
  h -= 1;
  w -= 1;

  T idx_x = at::arange(1, w, options).view({1,w-1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_y = at::arange(1, h, options).view({1,h-1, 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d-1, 1 , 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  }

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d-1,h-1,w-1});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_x = 0.25 * (self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z-1,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y  ,idx_x+1}) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z-1,idx_y  ,idx_x+1}) );

  T v_y = 0.25 * (self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z-1,idx_y  ,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y+1,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z-1,idx_y+1,idx_x  }) );
  T v_z = at::zeros_like(v_x);

  if (is3D) {
    v_z = self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x  });
  }
  return at::stack({v_x, v_y, v_z}, 1);


}

T interpolComponent(const T& self, const T& pos, int c) {

  AT_ASSERTM(pos.size(1) == 3, "Input pos must have 3 channels");

  bool is3D = (self.size(2) > 1);
  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);

  auto options = self.options();

  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T pos0 = p.toType(at::kLong);


  T s1 = p.select(1,0) - pos0.select(1,0).toType(self.scalar_type());
  T t1 = p.select(1,1) - pos0.select(1,1).toType(self.scalar_type());
  T f1 = p.select(1,2) - pos0.select(1,2).toType(self.scalar_type());
  T s0 = 1 - s1;
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  T x0 = pos0.select(1,0).clamp_(0, self.size(4) - 2);
  T y0 = pos0.select(1,1).clamp_(0, self.size(3) - 2);
  T z0 = pos0.select(1,2).clamp_(0, self.size(2) - 2);

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(x0.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(pos0.scalar_type());

  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);




  if (is3D) {
   T Ia= self.index({idx_b, idx_c.select(1,c), z0  , y0  , x0  });
   T Ib= self.index({idx_b, idx_c.select(1,c), z0  , y0+1, x0  });
   T Ic= self.index({idx_b, idx_c.select(1,c), z0  , y0  , x0+1});
   T Id= self.index({idx_b, idx_c.select(1,c), z0  , y0+1, x0+1});
   T Ie= self.index({idx_b, idx_c.select(1,c), z0+1, y0  , x0  });
   T If= self.index({idx_b, idx_c.select(1,c), z0+1, y0+1, x0  });
   T Ig= self.index({idx_b, idx_c.select(1,c), z0+1, y0  , x0+1});
   T Ih= self.index({idx_b, idx_c.select(1,c), z0+1, y0+1, x0+1});


    return ( (((Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1)*f0 +
             ((Ie*t0 + If*t1)*s0 + (Ig*t0 + Ih*t1)*s1)*f1 ).unsqueeze(1) );
  } else {
   T Ia= self.index({idx_b, idx_c.select(1,c), z0  , y0  , x0  });
   T Ib= self.index({idx_b, idx_c.select(1,c), z0  , y0+1, x0  });
   T Ic= self.index({idx_b, idx_c.select(1,c), z0  , y0  , x0+1});
   T Id= self.index({idx_b, idx_c.select(1,c), z0  , y0+1, x0+1});

    return ( ((Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1 ).unsqueeze(1) );
  }
}


T interpolComponent_temp(const T& self, const T& pos, int c) {

  AT_ASSERTM(pos.size(1) == 3, "Input pos must have 3 channels");

  bool is3D = (self.size(2) > 1);
  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);

  auto options = self.options();

  // 0.5 is defined as the center of the first cell as the scheme shows:
  //   |----x----|----x----|----x----|
  //  x=0  0.5   1   1.5   2   2.5   3
  T p = pos - 0.5;

  // Cast to integer, truncates towards 0.
  T pos0 = p.toType(at::kLong);

  T s1 = p.select(1,0) - pos0.select(1,0).toType(self.scalar_type());
  T t1 = p.select(1,1) - pos0.select(1,1).toType(self.scalar_type());
  T f1 = p.select(1,2) - pos0.select(1,2).toType(self.scalar_type());
  T s0 = 1 - s1;
  T t0 = 1 - t1;
  T f0 = 1 - f1;

  T x0 = pos0.select(1,0);
  T y0 = pos0.select(1,1);
  T z0 = pos0.select(1,2);

  // We avoid the clamping so that if we go interpolating
  // to the right we go to the value on the left (-1)
  T mns_one = at::ones_like(x0); // Floating zero

  T x_new = x0.where(x0<self.size(4) -1,(x0-(self.size(4))*mns_one));
  T y_new = y0.where(y0<self.size(3) -1,(y0-(self.size(3))*mns_one));
  T z_new = z0.where(z0<self.size(2),(z0-(self.size(2))*mns_one));

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(x0.scalar_type());
  idx_b = idx_b.expand({bsz,d,h,w});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(pos0.scalar_type());

  s1.clamp_(0, 1);
  t1.clamp_(0, 1);
  f1.clamp_(0, 1);
  s0.clamp_(0, 1);
  t0.clamp_(0, 1);
  f0.clamp_(0, 1);


  if (is3D) {
   T Ia= self.index({idx_b, idx_c.select(1,c), z0  , y0  , x0  });
   T Ib= self.index({idx_b, idx_c.select(1,c), z0  , y0+1, x0  });
   T Ic= self.index({idx_b, idx_c.select(1,c), z0  , y0  , x0+1});
   T Id= self.index({idx_b, idx_c.select(1,c), z0  , y0+1, x0+1});
   T Ie= self.index({idx_b, idx_c.select(1,c), z0+1, y0  , x0  });
   T If= self.index({idx_b, idx_c.select(1,c), z0+1, y0+1, x0  });
   T Ig= self.index({idx_b, idx_c.select(1,c), z0+1, y0  , x0+1});
   T Ih= self.index({idx_b, idx_c.select(1,c), z0+1, y0+1, x0+1});

    return ( (((Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1)*f0 +
             ((Ie*t0 + If*t1)*s0 + (Ig*t0 + Ih*t1)*s1)*f1 ).unsqueeze(1) );
  } else {

   T Ia= self.index({idx_b, idx_c.select(1,c), z0  , y_new  , x_new  });
   T Ib= self.index({idx_b, idx_c.select(1,c), z0  , y_new+1, x_new  });
   T Ic= self.index({idx_b, idx_c.select(1,c), z0  , y_new  , x_new+1});
   T Id= self.index({idx_b, idx_c.select(1,c), z0  , y_new+1, x_new+1});


    return ( ((Ia*t0 + Ib*t1)*s0 + (Ic*t0 + Id*t1)*s1 ).unsqueeze(1) );
  }
}

// End of new interpol

T curl(const T& self) {
  AT_ASSERTM(self.size(1) == 3, "Input velocity field must have 3 channels");

  int bsz = self.size(0);
  int d = self.size(2) ;
  int h = self.size(3);
  int w = self.size(4);

  auto options = self.options();

  bool is3D = (d > 1);
  d = is3D? (d-1): 2;
  h -= 1;
  w -= 1;

  T idx_x = at::arange(1, w, options).view({1,w-1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_y = at::arange(1, h, options).view({1,h-1, 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  T idx_z = at::zeros_like(idx_x);
  if (is3D) {
     idx_z = at::arange(1, d, options).view({1,d-1, 1 , 1}).expand({bsz,d-1,h-1,w-1}).toType(at::kLong);
  }

  T idx_b = at::arange(0, bsz, options).view({bsz,1,1,1}).toType(idx_x.scalar_type());
  idx_b = idx_b.expand({bsz,d-1,h-1,w-1});

  T idx_c = at::arange(0, 3, options).view({1,3,1,1,1}).toType(idx_x.scalar_type());

  T v_z = 0.5 *  (self.index({idx_b,idx_c.select(1,1) ,idx_z  ,idx_y  ,idx_x+1}) +
                  self.index({idx_b,idx_c.select(1,1) ,idx_z-1,idx_y  ,idx_x-1}) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y+1,idx_x  }) +
                  self.index({idx_b,idx_c.select(1,0) ,idx_z  ,idx_y-1,idx_x  }) );

  T v_y = at::zeros_like(v_z);
  T v_x = at::zeros_like(v_z);

  if (is3D) {
    v_x = 0.5 * (self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y+1,idx_x  }) +
                 self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y-1,idx_x  }) +
                 self.index({idx_b,idx_c.select(1,1) ,idx_z+1,idx_y  ,idx_x  }) +
                 self.index({idx_b,idx_c.select(1,1) ,idx_z-1,idx_y  ,idx_x  }) );

    v_y = 0.5 * (self.index({idx_b,idx_c.select(1,0) ,idx_z+1,idx_y  ,idx_x  }) +
                 self.index({idx_b,idx_c.select(1,0) ,idx_z-1,idx_y  ,idx_x  }) +
                 self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x+1}) +
                 self.index({idx_b,idx_c.select(1,2) ,idx_z  ,idx_y  ,idx_x-1}) );
  }

  return at::stack({v_x, v_y, v_z}, 1);
}

} // namespace fluid
