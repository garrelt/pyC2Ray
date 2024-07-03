#ifndef RAYTRACING_HIP_H
#define RAYTRACING_HIP_H

#include "memory.hip.h"
#include "rates.hip.h"
#include <exception>
#include <string>
#include <iostream>

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define FOURPI 12.566370614359172463991853874177    // 4π
#define INV4PI 0.079577471545947672804111050482     // 1/4π
#define SQRT3 1.73205080757                         // Square root of 3
#define MAX_COLDENSH 2e30                           // Column density limit (rates are set to zero above this)
#define HIP_BLOCK_SIZE 256                         // Size of blocks used to treat sources

// ========================================================================
// Utility Device Functions
// ========================================================================

// Fortran-type modulo function (C modulo is signed)
inline int modulo(const int & a,const int & b) { return (a%b+b)%b; }
inline __device__ int modulo_gpu(const int & a,const int & b) { return (a%b+b)%b; }

// Sign function on the device
inline __device__ int sign_gpu(const double & x) { if (x>=0) return 1; else return -1;}

// Flat-array index from 3D (i,j,k) indices
inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N) { return N*N*i + N*j + k;}

// Weight function for C2Ray interpolation function (see cinterp_gpu below)
__device__ inline double weightf_gpu(const double & cd, const double & sig) { return 1.0/max(0.6,cd*sig);}

// Mapping from cartesian coordinates of a cell to reduced cache memory space (here N = 2qmax + 1 in general)
__device__ inline int cart2cache(const int & i,const int & j,const int & k,const int & N) { return N*N*int(k<0) + N*i + j; }

// Mapping from linear 1D indices to the cartesian coords of a q-shell in asora
__device__ void linthrd2cart(const int & s,const int & q,int& i,int& j)
{
    if (s == 0)
    {
        i = q;
        j = 0;
    }
    else
    {
        int b = (s - 1) / (2*q);
        int a = (s - 1) % (2*q);

        if (a + 2*b > 2*q)
        {
            a = a + 1;
            b = b - 1 - q;
        }
        i = a + b - q;
        j = b;
    }
}

// When using a GPU with compute capability < 6.0, we must manually define the atomicAdd function for doubles
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ========================================================================
// Raytrace all sources and add up ionization rates
// ========================================================================
void do_all_sources_gpu(
    const double & R,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    double* ndens,
    double* xh_av,
    double* phi_ion,
    const int & NumSrc,
    const int & m1,
    const double & minlogtau,
    const double & dlogtau,
    const int & NumTau)
{
    // Byte-size of grid data
    int meshsize = m1*m1*m1*sizeof(double);

    // Determine how large the octahedron should be, based on the raytracing radius. Currently,
    // this is set s.t. the radius equals the distance from the source to the middle of the faces
    // of the octahedron. To raytrace the whole box, the octahedron must be 1.5*N in size
    int max_q = std::ceil(SQRT3 * min(R,SQRT3*m1/2.0));

    // HIP Grid size: since 1 block = 1 source, this sets the number of sources treated in parallel
    dim3 gs(NUM_SRC_PAR);

    // HIP Block size: more of a tuning parameter (see above), in practice anything ~128 is fine
    dim3 bs(HIP_BLOCK_SIZE);

    // Here we fill the ionization rate array with zero before raytracing all sources. The LOCALRATES flag
    // is for debugging purposes and will be removed later on
    hipMemset(phi_dev,0,meshsize);

    // Copy current ionization fraction to the device
    hipMemcpy(x_dev,xh_av,meshsize,hipMemcpyHostToDevice);

    // Since the grid is periodic, we limit the maximum size of the raytraced region to a cube as large as the mesh around the source.
    // See line 93 of evolve_source in C2Ray, this size will depend on if the mesh is even or odd.
    // Basically the idea is that you never touch a cell which is outside a cube of length ~N centered on the source
    int last_r = m1/2 - 1 + modulo(m1,2);
    int last_l = -m1/2;

    // Loop over batches of sources
    for (int ns = 0; ns < NumSrc; ns += NUM_SRC_PAR)
    {
        // Raytrace the current batch of sources in parallel
        evolve0D_gpu<<<gs,bs>>>(R,max_q,ns,NumSrc,NUM_SRC_PAR,src_pos_dev,src_flux_dev,cdh_dev,
            sig,dr,n_dev,x_dev,phi_dev,m1,photo_thin_table_dev,photo_thick_table_dev,
            minlogtau,dlogtau,NumTau,last_l,last_r);

        // Check for errors
        auto error = hipGetLastError();
        if(error != hipSuccess) {
            throw std::runtime_error("Error Launching Kernel: "
                                    + std::string(hipGetErrorName(error)) + " - "
                                    + std::string(hipGetErrorString(error)));
        }

        // Sync device to be sure (is this required ??)
        hipDeviceSynchronize();
    }

    // Copy the accumulated ionization fraction back to the host
    auto error = hipMemcpy(phi_ion,phi_dev,meshsize,hipMemcpyDeviceToHost);
}

// ========================================================================
// Raytracing kernel, adapted from C2Ray. Calculates in/out column density
// to the current cell and finds the photoionization rate
// ========================================================================
__global__ void evolve0D_gpu(
    const double Rmax_LLS,
    const int q_max,    // Is now the size of max q
    const int ns_start,
    const int NumSrc,
    const int num_src_par,
    int* src_pos,
    double* src_flux,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1,
    const double* photo_thin_table,
    const double* photo_thick_table,
    const double minlogtau,
    const double dlogtau,
    const int NumTau,
    const int last_l,
    const int last_r
)
{
    /* The raytracing kernel proceeds as follows:
    1. Select the source based on the block number (within the batch = the grid)
    2. Loop over the asora q-cells around the source, up to q_max (loop "A")
    3. Inside each shell, threads independently do all cells, possibly requiring multiple iterations
    if the block size is smaller than the number of cells in the shell (loop "B")
    4. After each shell, the threads are synchronized to ensure that causality is respected
    */

    // Source number = Start of batch + block number (each block does one source)
    int ns = ns_start + blockIdx.x;

    // Offset pointer to the outgoing column density array used for interpolation (each block
    // needs its own copy of the array)
    int cdh_offset = blockIdx.x * m1 * m1 * m1;

    // Ensure the source index is valid
    if (ns < NumSrc)
    {
        // (A) Loop over ASORA q-shells
        for (int q = 0 ; q <= q_max ; q++)
        {
            // We figure out the number of cells in the shell and determine how many passes
            // the block needs to take to treat all of them
            int num_cells = 4*q*q + 2;
            int Npass = num_cells / HIP_BLOCK_SIZE + ((num_cells % HIP_BLOCK_SIZE) ? 1 : 0);

            // Iterate over each shell
            for (int pass = 0; pass < Npass; pass++)
            {
                // Thread index within this kernel call
                int id = threadIdx.x + blockDim.x * pass;

                if (id < num_cells)
                {
                    // Map cell to (i,j,k)
                    int i, j;
                    linthrd2cart(id, q, i, j);

                    // Get the source position and calculate the column density
                    int src_idx = ns + src_pos[0];
                    double src_x = src_pos[1];
                    double src_y = src_pos[2];
                    double src_z = src_pos[3];

                    // Compute column density and ionization rate
                    double r = sqrt(pow(src_x - i, 2) + pow(src_y - j, 2) + pow(src_z - j, 2));
                    double tau = min(log10(r) / dlogtau, NumTau - 1);
                    tau = minlogtau + dlogtau * tau;

                    // Perform column density and photoionization rate computations
                    double* cdh = coldensh_out + cdh_offset;
                    double* flux = src_flux + ns;
                    if (tau >= 0)
                    {
                        double cd = cdh[id];
                        double rate = flux[0] * rate_function(cd, sig, photo_thin_table, photo_thick_table, minlogtau, dlogtau, NumTau);
                        atomicAdd(phi_ion + mem_offst_gpu(i, j, k, m1), rate);
                    }
                }
            }
        }
    }
}

#endif // RAYTRACING_HIP_H
