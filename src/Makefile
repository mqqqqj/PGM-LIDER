test: test.cpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 test.cpp -I ../include -o test_core.out

lider: lider.cpp ../include/lider/lider.hpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 lider.cpp -I ../include -o lider.out

hili: hili.cpp ../include/lider/hili.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 hili.cpp -I ../include -o hili.out

runtest: test
	nohup ./test_core.out > test_core_result.txt 2>&1 &

runlider: lider
	nohup ./lider.out > lider_result.txt 2>&1 &

runhili: hili
	nohup ./hili.out > hili_result.txt 2>&1 &

perf: lider
	perf record -g ./lider.out && perf report > ./lider_report.txt

clean:
	rm -f *.out *.data *.data.old *_report.txt massif.out.* *_result.txt