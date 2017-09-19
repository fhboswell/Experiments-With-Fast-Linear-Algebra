
BITCODE += ${CL_SOURCES:.cl=.cpu32.bc}
BITCODE += ${CL_SOURCES:.cl=.cpu64.bc}
BITCODE += ${CL_SOURCES:.cl=.gpu32.bc}
BITCODE += ${CL_SOURCES:.cl=.gpu64.bc}

OBJECTS := ${SOURCES:.c=.o}

$(TARGET): $(BITCODE) $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(FRAMEWORKS) $(ARCHS)

%.o: %.c
	$(CC) $(CFLAGS) $(ARCHS) $< -o $@

# #
# The OpenCL C compilation commands for 32/64bit CPUs and GPUs:
#
# As an example, to compile for a 32bit GPU:
# openclc -emit-llvm -c -arch gpu_32 kernel.cl -o kernel.bc
# #
%.cpu32.bc: %.cl
	$(CLC) -emit-llvm -c -arch i386 $< -o $@

%.cpu64.bc: %.cl
	$(CLC) -emit-llvm -c -arch x86_64 $< -o $@

%.gpu32.bc: %.cl
	$(CLC) -emit-llvm -c -arch gpu_32 $< -o $@

%.gpu64.bc: %.cl
	$(CLC) -emit-llvm -c -arch gpu_64 $< -o $@

clean:
	rm -rf $(TARGET) $(BITCODE) $(OBJECTS)
