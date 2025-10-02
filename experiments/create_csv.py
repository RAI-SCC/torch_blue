import csv

with open('./outputs/scaling.csv', 'w', newline='') as csvfile:
    fieldnames = ['dataset', 'architecture', 'parallel_strategy', 'sample_num', 'gpu_num', 'epoch_time', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()