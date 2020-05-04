import pandas as pd
from verzoekveld_functies import *


def test_opschonen_oplossingen():
    oplossingen_input = pd.read_excel("./data/test/Standaardoplossingen_test_1.xlsx")
    oplossingen_check = pd.read_excel("./data/test/Standaardoplossingen_test_1_check.xlsx")
    oplossingen_input = opschonen_oplossingen(oplossingen_input)
    oplossingen_input.reset_index(drop=True, inplace=True)
    oplossingen_check.reset_index(drop=True, inplace=True)
    oplossingen_input = oplossingen_input.\
            reindex(sorted(oplossingen_input.columns),
                    axis = 1)
    oplossingen_check = oplossingen_check.\
            reindex(sorted(oplossingen_check.columns),
                    axis = 1)

    assert oplossingen_input.equals(oplossingen_check)

def test_splitsen_oplossingen():
    oplossingen_input = pd.read_excel("./data/test/Standaardoplossingen_test_2.xlsx")
    oplossingen_check = pd.read_excel("./data/test/Standaardoplossingen_test_2_check.xlsx")
    oplossingen_input = splits_oplossingen(oplossingen_input)
    oplossingen_check['Verzoek_regex'] =\
            oplossingen_check['Verzoek_regex'].fillna("")
    oplossingen_input = oplossingen_input.replace(to_replace=[None], 
            value=np.nan)
    print(oplossingen_input.columns.values)
    print(oplossingen_input.columns.values)
    
    print(oplossingen_input[['Verzoek_regex']])
    for 
    print(oplossingen_check[['Verzoek_regex']])
    assert oplossingen_input.equals(oplossingen_check)
    assert oplossingen_input['Verzoek_regex'].equals(oplossingen_check['Verzoek_regex'])


    



