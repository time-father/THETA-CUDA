#ifndef THETA_EXPONENTIALS_H
#define THETA_EXPONENTIALS_H

#include "cuh/cuda_functions.cuh"
namespace theta {
// returns 1 - exp(-tau)
// Equivalent to -(_expm1f(-tau)), but faster
// Written by Colin Josey.
// double version
#ifndef THETA_CUDA
#endif
HDD inline double double_cjosey_exponential(double tau)
{
  constexpr double c1n = -1.0000013559236386308f;
  constexpr double c2n = 0.23151368626911062025f;
  constexpr double c3n = -0.061481916409314966140f;
  constexpr double c4n = 0.0098619906458127653020f;
  constexpr double c5n = -0.0012629460503540849940f;
  constexpr double c6n = 0.00010360973791574984608f;
  constexpr double c7n = -0.000013276571933735820960f;

  constexpr double c0d = 1.0f;
  constexpr double c1d = -0.73151337729389001396f;
  constexpr double c2d = 0.26058381273536471371f;
  constexpr double c3d = -0.059892419041316836940f;
  constexpr double c4d = 0.0099070188241094279067f;
  constexpr double c5d = -0.0012623388962473160860f;
  constexpr double c6d = 0.00010361277635498731388f;
  constexpr double c7d = -0.000013276569500666698498f;

  double x = -tau;

  double den = c7d;
  den = den * x + c6d;
  den = den * x + c5d;
  den = den * x + c4d;
  den = den * x + c3d;
  den = den * x + c2d;
  den = den * x + c1d;
  den = den * x + c0d;

  double num = c7n;
  num = num * x + c6n;
  num = num * x + c5n;
  num = num * x + c4n;
  num = num * x + c3n;
  num = num * x + c2n;
  num = num * x + c1n;
  num = num * x;

  return num / den;
}

// float version
HDD inline float cjosey_exponential(float tau)
{
  constexpr float c1n = -1.0000013559236386308f;
  constexpr float c2n = 0.23151368626911062025f;
  constexpr float c3n = -0.061481916409314966140f;
  constexpr float c4n = 0.0098619906458127653020f;
  constexpr float c5n = -0.0012629460503540849940f;
  constexpr float c6n = 0.00010360973791574984608f;
  constexpr float c7n = -0.000013276571933735820960f;

  constexpr float c0d = 1.0f;
  constexpr float c1d = -0.73151337729389001396f;
  constexpr float c2d = 0.26058381273536471371f;
  constexpr float c3d = -0.059892419041316836940f;
  constexpr float c4d = 0.0099070188241094279067f;
  constexpr float c5d = -0.0012623388962473160860f;
  constexpr float c6d = 0.00010361277635498731388f;
  constexpr float c7d = -0.000013276569500666698498f;

  float x = -tau;

  float den = c7d;
  den = den * x + c6d;
  den = den * x + c5d;
  den = den * x + c4d;
  den = den * x + c3d;
  den = den * x + c2d;
  den = den * x + c1d;
  den = den * x + c0d;

  float num = c7n;
  num = num * x + c6n;
  num = num * x + c5n;
  num = num * x + c4n;
  num = num * x + c3n;
  num = num * x + c2n;
  num = num * x + c1n;
  num = num * x;

  return num / den;
}

// The below two functions (exponentialG and exponentialG2) were developed
// by Colin Josey. The implementation of these functions is closely based
// on the OpenMOC versions of these functions. The OpenMOC license is given
// below:

// Copyright (C) 2012-2023 Massachusetts Institute of Technology and OpenMOC
// contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Computes y = 1/x-(1-exp(-x))/x**2 using a 5/6th order rational
// approximation. It is accurate to 2e-7 over [0, 1e5]. Developed by Colin
// Josey using Remez's algorithm, with original implementation in OpenMOC at:
// https://github.com/mit-crpg/OpenMOC/blob/develop/src/exponentials.h

// float version
inline float exponentialG(float tau)
{
  // Numerator coefficients in rational approximation for 1/x - (1 - exp(-x)) /
  // x^2
  constexpr float d0n = 0.5f;
  constexpr float d1n = 0.176558112351595f;
  constexpr float d2n = 0.04041584305811143f;
  constexpr float d3n = 0.006178333902037397f;
  constexpr float d4n = 0.0006429894635552992f;
  constexpr float d5n = 0.00006064409107557148f;

  // Denominator coefficients in rational approximation for 1/x - (1 - exp(-x))
  // / x^2
  constexpr float d0d = 1.0f;
  constexpr float d1d = 0.6864462055546078f;
  constexpr float d2d = 0.2263358514260129f;
  constexpr float d3d = 0.04721469893686252f;
  constexpr float d4d = 0.006883236664917246f;
  constexpr float d5d = 0.0007036272419147752f;
  constexpr float d6d = 0.00006064409107557148f;

  float x = tau;

  float num = d5n;
  num = num * x + d4n;
  num = num * x + d3n;
  num = num * x + d2n;
  num = num * x + d1n;
  num = num * x + d0n;

  float den = d6d;
  den = den * x + d5d;
  den = den * x + d4d;
  den = den * x + d3d;
  den = den * x + d2d;
  den = den * x + d1d;
  den = den * x + d0d;

  return num / den;
}

// Computes y = 1/x-(1-exp(-x))/x**2 using a 5/6th order rational
// approximation. It is accurate to 2e-7 over [0, 1e5]. Developed by Colin
// Josey using Remez's algorithm, with original implementation in OpenMOC at:
// https://github.com/mit-crpg/OpenMOC/blob/develop/src/exponentials.h
// double version
inline double exponentialG(double tau)
{
  // Numerator coefficients in rational approximation for 1/x - (1 - exp(-x)) /
  // x^2
  constexpr double d0n = 0.5;
  constexpr double d1n = 0.176558112351595;
  constexpr double d2n = 0.04041584305811143;
  constexpr double d3n = 0.006178333902037397;
  constexpr double d4n = 0.0006429894635552992;
  constexpr double d5n = 0.00006064409107557148;

  // Denominator coefficients in rational approximation for 1/x - (1 - exp(-x))
  // / x^2
  constexpr double d0d = 1.0;
  constexpr double d1d = 0.6864462055546078;
  constexpr double d2d = 0.2263358514260129;
  constexpr double d3d = 0.04721469893686252;
  constexpr double d4d = 0.006883236664917246;
  constexpr double d5d = 0.0007036272419147752;
  constexpr double d6d = 0.00006064409107557148;

  double x = tau;

  double num = d5n;
  num = num * x + d4n;
  num = num * x + d3n;
  num = num * x + d2n;
  num = num * x + d1n;
  num = num * x + d0n;

  double den = d6d;
  den = den * x + d5d;
  den = den * x + d4d;
  den = den * x + d3d;
  den = den * x + d2d;
  den = den * x + d1d;
  den = den * x + d0d;

  return num / den;
}

// Computes G2 : y = 2/3 - (1 + 2/x) * (1/x + 0.5 - (1 + 1/x) * (1-exp(-x)) /
// x) using a 5/5th order rational approximation. It is accurate to 1e-6 over
// [0, 1e6]. Developed by Colin Josey using Remez's algorithm, with original
// implementation in OpenMOC at:
// https://github.com/mit-crpg/OpenMOC/blob/develop/src/exponentials.h
// float version
inline float exponentialG2(float tau)
{

  // Coefficients for numerator in rational approximation
  constexpr float g1n = -0.08335775885589858f;
  constexpr float g2n = -0.003603942303847604f;
  constexpr float g3n = 0.0037673183263550827f;
  constexpr float g4n = 0.00001124183494990467f;
  constexpr float g5n = 0.00016837426505799449f;

  // Coefficients for denominator in rational approximation
  constexpr float g1d = 0.7454048371823628f;
  constexpr float g2d = 0.23794300531408347f;
  constexpr float g3d = 0.05367250964303789f;
  constexpr float g4d = 0.006125197988351906f;
  constexpr float g5d = 0.0010102514456857377f;

  float x = tau;

  float num = g5n;
  num = num * x + g4n;
  num = num * x + g3n;
  num = num * x + g2n;
  num = num * x + g1n;
  num = num * x;

  float den = g5d;
  den = den * x + g4d;
  den = den * x + g3d;
  den = den * x + g2d;
  den = den * x + g1d;
  den = den * x + 1.0f;

  return num / den;
}

// double version
inline double exponentialG2(double tau)
{

  // Coefficients for numerator in rational approximation
  constexpr double g1n = -0.08335775885589858;
  constexpr double g2n = -0.003603942303847604;
  constexpr double g3n = 0.0037673183263550827;
  constexpr double g4n = 0.00001124183494990467;
  constexpr double g5n = 0.00016837426505799449;

  // Coefficients for denominator in rational approximation
  constexpr double g1d = 0.7454048371823628;
  constexpr double g2d = 0.23794300531408347;
  constexpr double g3d = 0.05367250964303789;
  constexpr double g4d = 0.006125197988351906;
  constexpr double g5d = 0.0010102514456857377;

  double x = tau;

  double num = g5n;
  num = num * x + g4n;
  num = num * x + g3n;
  num = num * x + g2n;
  num = num * x + g1n;
  num = num * x;

  double den = g5d;
  den = den * x + g4d;
  den = den * x + g3d;
  den = den * x + g2d;
  den = den * x + g1d;
  den = den * x + 1.0;

  return num / den;
}

} // namespace theta
#endif // THETA_EXPONENTIALS_H
