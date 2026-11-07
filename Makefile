.PHONY: build play smoke
build:
	cmake -S . -B build
	cmake --build build --parallel
play: build
	./build/hashgrid_demo
smoke: build
	./build/hashgrid_demo --headless --frames 10 --res 128
