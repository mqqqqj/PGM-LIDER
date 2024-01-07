test: test.cpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 test.cpp -I ../include -o test_core && ./test_core

lider: lider.cpp ../include/lider/lider.hpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 lider.cpp -I ../include -o lider && ./lider

run: lider
	./lider

clean:
	rm -f lider test_core *.out