test: test.cpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -fopenmp -mavx2 -std=c++17 test.cpp -I ../include -o test.out

lider: lider.cpp ../include/lider/lider.hpp ../include/lider/core_model.hpp ../include/lider/utils.hpp
	g++ -fopenmp -mavx2 -std=c++17 lider.cpp -I ../include -o lider.out
# 调试时:g++ -g -gdwarf-4 -fopenmp -mavx2 -std=c++17 lider.cpp -I ../include -o lider.out
hili: hili.cpp ../include/lider/hili.hpp ../include/lider/utils.hpp
	g++ -g -fopenmp -std=c++17 hili.cpp -I ../include -o hili.out

runtest: test
	nohup ./test.out > test_result.txt 2>&1 &

runlider: lider
	nohup ./lider.out > lider_result.txt 2>&1 &

runhili: hili
	nohup ./hili.out > hili_result.txt 2>&1 &

gprof: runlider
	gprof lider.out > lider_profile.txt

clean:
	rm -f *.out *.data *.data.old *_report.txt massif.out.* *_result.txt