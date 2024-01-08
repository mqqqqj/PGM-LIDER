test: test.cpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 test.cpp -I ../include -o test_core.out

lider: lider.cpp ../include/lider/lider.hpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 lider.cpp -I ../include -o lider.out

runtest: test
	./test_core.out

runlider: lider
	./lider.out

perf: lider
	perf record -g ./lider.out && perf report > ./lider_report.txt

clean:
	rm -f *.out *.data *.data.old *_report.txt massif.out.*