CFLAGS=-Wall 
LIBS=-O2 -L/usr/X11R6/lib -lm -lpthread -lX11
EXE=SuperpixelHierarchy

all:
	mkdir -p res
	g++ $(CFLAGS) SuperpixelHierarchy_c.cpp -o $(EXE) $(LIBS)

test:
	./SuperpixelHierarchy -i ./data/335094.jpg  -c ./data/335094_edge.png
