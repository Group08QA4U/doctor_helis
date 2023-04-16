nohup python bayes.py -q 2 -width 20000 -height 20000 -p 14 -a 14 -r 40 -d 14 &> bayes.out.q2.w20000.h20000.p14.a14.r40.d14 &&
python visualize_bayes_result.py -i bayes.out.q2.w20000.h20000.p14.a14.r40.d14
