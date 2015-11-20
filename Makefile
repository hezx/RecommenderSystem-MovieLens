# makefile for stat640 movielens project #

VERSION:= 1.0.0
SRCDIR:= ./src/
CC:= gcc		# The C compiler.
CXXPG:= g++ -pg
CXX:= g++ -g -std=c++0x
CXXFLAGS:= -O3 -Wall 
d:=0

INCLUDEDIR:= -I/usr/include -I./include/ -I./src/ 
LIBDIR:= -L/usr/lib #-L./libs/
LIBS:= -lm  #-lgsl -lgslcblas # -ljson_linux-gcc-4.5.2_libmt

JSON:= src/lib_json/json_reader.cpp src/lib_json/json_value.cpp  src/lib_json/json_writer.cpp
JOBJS:= $(JSON:.cpp=.o)

help :
	@echo " "
	@echo "MovieLens version $(VERSION) source code"
	@echo " "
	@echo "Type ...          To ..."
	@echo "make all        	 Compile the MovieLens program"
	@echo "make small      	 Compile the MovieLens_Debug program"
	@echo "make clean        Delete temporary files and all .o objects"
	@echo "make clear        Deep clean"
	@echo " "

SRC:= ./src/Argument_helper.cpp ./src/main.cpp ./src/basic.cpp ./src/cross_validation.cpp ./src/baseline.cpp ./src/neighborhood.cpp ./src/svd.cpp ./src/svdasym.cpp ./src/svdplusplus.cpp ./src/svdneighbor.cpp
OBJS:= $(SRC:.cpp=.o)

EXE:= MovieLens
mlens: $(EXE)
$(EXE): $(OBJS) $(JOBJS) 
	$(CXX) $(INCLUDEDIR) $(OBJS) $(JOBJS) -o $(EXE) $(LIBDIR) $(LIBS) 

.cpp.o: $*.cpp
	$(CXX) -D SMALLDATA=$(d) $(CXXFLAGS) $(INCLUDEDIR) -c -o $@  $<
.c.o: $*.c
	$(CXX)  $(CXXFLAGS) $(INCLUDEDIR) -c -o $@  $<


.SUFFIXES : .cpp .c .o $(SUFFIXES)

clean:
	rm -rf $(SRCDIR)*.o $(SRCDIR)*~
clear:
	find $(SRCDIR) -type f -name "*.o" -delete
all:
	make mlens d=0
small:
	make clean mlens d=1
elf:
	cd blend; make; make clean; mv ELF ../; cd ..
