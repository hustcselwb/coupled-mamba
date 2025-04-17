from train_test import train,prepare_trte_data
from mm_model import MMDynamic
if __name__ == "__main__":
    data_folder="/root/mamba-main/BRCA/BRCA"
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder, use_view=['mrna','dna', 'mirna'])
    print(data_tr_list[2].shape)
    print(len(data_tr_list))
    
    print(len(data_test_list))
    print(data_test_list[0].shape)
    print(len(trte_idx))
    print(type(trte_idx))
    print(trte_idx['tr'])
    print(trte_idx['te'])


