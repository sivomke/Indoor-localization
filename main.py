import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# find coordinates of a point in N-dim space with Euclidean metric
# given distances to m points with known coordinates

# input data frame
df = pd.read_csv('iBeacon_RSSI_Labeled.csv')
# print(df.head())
# print(df.tail())
# df.info()
# print(df.shape)
print(df.keys().to_list())
beacons_name = np.array(df.keys().to_list()[2:])


# as we newest measurements are on top
# as we want to reconstruct trajectory we want to start from earlier measurements
df = df[::-1]  # we iterate through rows in reverse order

# first, as it makes sense to build trajectory for particular day, we split 'data' column into 'date' and 'time' columns
split = df['date'].str.split().to_list()  # list of lists containing date and time
date_data = [sublist[0] for sublist in split]
time_data = [sublist[1] for sublist in split]

df.drop(columns='date', inplace=True)  # delete existing 'date' column
df.insert(loc=1, column='date', value=date_data)
df.insert(loc=2, column='time', value=time_data)

# list of distinct dates on which the measurements were done
dates = df['date'].drop_duplicates().to_list()
print("# of dates: {}".format(len(dates)))
print(dates)

current_date = dates[2]
print("current date: {}".format(current_date))
current_df = df[df['date'] == current_date]  # extract date corresponding to current date
# print(current_df.head())
print("# of time points corresponding to chosen date: {}".format(current_df.shape[0]))

# now we want to extract measurements where we have RSSI from 3 or more iBeacons
selection = []  # i-th element is true iff row with the same index contains RSSI from 3 or more iBeacons
for i in range(current_df.shape[0]):
    num = (current_df.iloc[i, 3:] > -200).sum()  # num of RSSI >-200 for that measurement
    if num > 2:
        selection.append(True)
    else:
        selection.append(False)
    # print(current_df.iloc[i][3:].to_list())


# we create a 10*10 ft grid, with (0,0) being top-left corner

# approximate coordinates of 13 iBeacons in that grid
beacons_coords = np.array([[80, 50], [30, 100], [30, 140], [30, 180],
                           [65, 100], [65, 40], [65, 180], [95, 100],
                           [145, 40], [145, 100], [145, 100], [145, 180],
                           [145, 220]])

valid_df = current_df[selection]  # extracted measurements
print("# of valid measurements: {}".format(valid_df.shape[0]))
# print("1:  ".format(valid_df.iloc[0]))
# print(valid_df.iloc[0, 0])


# square of Euclidean distance in N-dim space

def distance(x, y) -> np.array:
    return np.dot(x-y, x-y)


def RSSI_to_dist(rssi, measured_power=-60, n=1.7):
    """

    :param rssi: given RSSI measurement;
    :param measured_power: constant depending on iBeacon type; expected RSSI at 1m distance
    :param n: constant that depends on environmental factor;
            typical values are: 2 for free space;
            2.7 to 3.5 for urban areas;
            3.0 to 5.0 in suburban areas;
            1.6 to 1.8 for indoors when there is line of sight to the router.
    :return: squared euclidean distance to that iBeacon
    """
    # return pow(10, 2*(measured_power - rssi)/(10*n))
    return (pow(10, (measured_power - rssi) / (10 * n)) * 3.2808)**2


# residual for i-th point
def r_i(i, x):
    return distance(x, valid_beacons[i]) - valid_dist[i]


# squared euclidean norm of error function
# we want to find argmin of f(x)
def f(x):
    return np.sum(np.array([r_i(i, x)**2 for i in range(m)]))


def dr_i_dx_j(x, i, j):
    return 2*(x[j] - valid_beacons[i, j])


def jacobian(x):
    return np.array([[dr_i_dx_j(x, i, j) for j in range(N)] for i in range(m)])


# function which computes the vector of residuals
def g(x):
    return np.array([r_i(i, x) for i in range(m)])


# custom implementation of Levenberg-Marquardt
def lm():
    x = np.random.rand(2)
    # print("initial guess: {}".format(x))
    steps = 0  # counts number of iterations until the solution is found
    damping_factor = .001  # regularization
    eye = np.identity(N)
    while True:
        J = jacobian(x)
        J_T = J.T
        inverse = np.linalg.inv(J_T @ J + damping_factor*eye)
        r = np.array([r_i(i, x) for i in range(m)])
        delta = (inverse @ J_T).dot(r)
        if f(x - delta) < f(x):
            if abs(np.dot(delta, delta)) < eps:
                break
            x = x - delta
            damping_factor *= 0.8
            # steps += 1
        else:
            damping_factor *= 2
        steps += 1
        # print("# {} iteration".format(steps))
        # print("damping factor: {:.6f}".format(damping_factor))
        # print("computed point: {}".format(x))
    return x


def find_index(x):
    return int(x//10)


def get_cell(coords: np.ndarray) -> str:
    """
    :param coords: position in the 10*10 ft grid, with (0,0) being top-left corner
    :return: position in the format of input data, ex. A01
    """
    letters = [chr(65 + i) for i in range(21)]
    rows = [str(i).zfill(2) for i in range(1, 19)]
    row_coord = coords[0]
    col_coord = coords[1]
    return letters[find_index(col_coord)]+rows[find_index(row_coord)]


for i in range(valid_df.shape[0]):
    RSSI = np.array(valid_df.iloc[i, 3:])
    valid_RSSI = np.array(RSSI[RSSI != -200])
    valid_beacons = np.array(beacons_coords[RSSI != -200])
    valid_names = beacons_name[RSSI != -200]
    print("seen by: {}".format(valid_names))


    m = valid_beacons.shape[0]

    N = 2  # dimension
    eps = pow(10, -8)

    valid_dist = [RSSI_to_dist(x) for x in valid_RSSI]

    # res_1 = least_squares(g, np.random.rand(2)).x
    res_2 = lm()
    # print(res_2)

    print("real: {}, calculated: {}".format(valid_df.iloc[i, 0], get_cell(res_2)))







