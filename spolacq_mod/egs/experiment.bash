cd ../scripts/

python main_simulate_wordseg.py --data_name= may-13_simulatedwordseg_10percent \
--simulated_wordseg={'num_words': 500, 'zero':5, 'one':5, 'two':5, 'three':5, 'four':5, 'five':5, 'six':5, 'seven':5, 'eight':5, 'nine':5}

python main_simulate_wordseg.py --data_name= may-13_simulatedwordseg_20percent \
--simulated_wordseg={'num_words': 500, 'zero':10, 'one':10, 'two':10, 'three':10, 'four':10, 'five':10, 'six':10, 'seven':10, 'eight':10, 'nine':10}

python main_simulate_wordseg.py --data_name= may-13_simulatedwordseg_30percent \
--simulated_wordseg={'num_words': 500, 'zero':15, 'one':15, 'two':15, 'three':15, 'four':15, 'five':15, 'six':15, 'seven':15, 'eight':15, 'nine':15}

python main_simulate_wordseg.py --data_name= may-13_simulatedwordseg_40percent \
--simulated_wordseg={'num_words': 500, 'zero':20, 'one':20, 'two':20, 'three':20, 'four':20, 'five':20, 'six':20, 'seven':20, 'eight':20, 'nine':20}
