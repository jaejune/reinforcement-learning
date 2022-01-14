import numpy as np
from gym import Env, utils
from terminaltables import SingleTable
from colorclass import Color, Windows
from gym import Env, utils
def main():
    test=  [ 
            "S 11 0 -1 H",
            "2 H 3 9 H",
            "10 29 H 2 1",
            "23 2 1 -3 H",
            "23 2 1 G 4",
            ]
    map = np.array([i.split(' ') for i in test])
    map = map.tolist()
    map[2][2] = utils.colorize(map[2][2], 'red', highlight=True)
    table_instance = SingleTable(map)
    table_instance.inner_heading_row_border = False
    table_instance.inner_row_border = True
    print(table_instance.table)

def table_server_status():
    """Return table string to be printed."""
    table_data = [
        [Color('Low Space'), Color('{autocyan}Nominal Space{/autocyan}'), Color('Excessive Space')],
        [Color('Low Load'), Color('Nominal Load'), Color('{autored}High Load{/autored}')],
        [Color('{autocyan}Low Free RAM{/autocyan}'), Color('Nominal Free RAM'), Color('High Free RAM')],
    ]
    table_instance = SingleTable(table_data, '192.168.0.105')
    table_instance.inner_heading_row_border = False
    table_instance.inner_row_border = True
    table_instance.justify_columns = {0: 'center', 2: 'center'}
    return table_instance.table
# print(table_server_status())
main()