test: test.cpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 test.cpp -I ../include -o test_core.out && ./test_core.out

lider: lider.cpp ../include/lider/lider.hpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 lider.cpp -I ../include -o lider.out && ./lider.out

run: lider
	./lider.out

clean:
	rm -f *.out