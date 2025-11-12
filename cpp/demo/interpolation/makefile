main: main.cpp
	nvc++ -cuda -O2 -pg -g -traceback -DHAS_PETSC  -DHAS_ADIOS2 -std=c++20 -isystem /usr/local/petsc/include  -isystem /usr/local/petsc/linux-gnu-real64-32-cuda/include \
-isystem /dolfinx-env/lib/python3.12/site-packages/ffcx/codegeneration \
-isystem /usr/local/include -L/usr/local/lib \
-isystem /cpp/dolfinx -L/cpp/build/dolfinx \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib  -L/usr/local/petsc/linux-gnu-real64-32-cuda/lib  main.cpp  -lbasix -ldolfinx -lmpi  -ladios2_cxx11 -lpetsc
