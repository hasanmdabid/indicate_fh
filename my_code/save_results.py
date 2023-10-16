import os
from pathlib import Path
from datetime import datetime
def saveResultsCSV(method, model_name, epochs, batch_size, avg_accuracy, std_avg_accuracy,  avg_f1_score_macro, std_f1_score_macro, max_f1_score_macro):
    path = './results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './results/results.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write(
            'Finished on; method, model_name, epochs, batch_size, avg_accuracy, std_avg_accuracy, avg_f1_score_macro, std_f1_score_macro,  max_f1_score_macro\n')
        f.close()
    with open(fileString, "a") as f:
        f.write('{};{};{};{};{};{};{};{};{};{}\n'.format(now, method, model_name, epochs, batch_size, avg_accuracy, std_avg_accuracy,  avg_f1_score_macro, std_f1_score_macro, 
                                                         max_f1_score_macro))
    f.close()