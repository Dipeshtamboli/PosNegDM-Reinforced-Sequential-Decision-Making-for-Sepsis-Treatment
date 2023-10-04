import pandas
import numpy as np

def get_accuracy(predicts, targets):
    predicts = np.argmax(predicts, axis=1)
    tot_num = len(predicts)
    correct_num = (predicts==targets).sum()
    accuracy = float(correct_num) / tot_num

    return accuracy

def read_dst(csv_path):
    # load
    df = pandas.read_csv(csv_path)
    # print(df)

    # preprocess
    observation_item = ['gender', 'mechvent', 'max_dose_vaso', 're_admission', 'age', 'Weight_kg', 'GCS', 'HR', 'SysBP',
                        'MeanBP', 'DiaBP', 'Temp_C', 'RR', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium',
                        'Calcium', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',
                        'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2',
                        'cumulated_balance', 'SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',
                        'input_total','input_4hourly','output_total','output_4hourly']

    # target: [{'rewards': , 'observations': , 'actions': , 'dones': }, ...]
    states = df[observation_item].values
    traj_idxes = df['traj'].values
    rewards = df['r'].values
    actions = df['a'].values
    
    print(states.shape, rewards.shape, actions.shape, traj_idxes.shape)

    trajectories = {}
    data_len = traj_idxes.shape[0]
    for i in range(data_len):
        idx = traj_idxes[i]
        if idx not in trajectories:
            trajectories[idx] = {'rewards': [], 'observations': [], 'actions': [], 'dones': []}

        trajectories[idx]['rewards'].append(float(rewards[i]))
        trajectories[idx]['actions'].append(int(actions[i]))
        trajectories[idx]['dones'].append(0)
        trajectories[idx]['observations'].append(states[i])

    # print(trajectories, list(trajectories.values()))
    trajectories = list(trajectories.values())
    # print(len(trajectories))
    # post process
    for traj in trajectories:
        for k in traj.keys():
            traj[k] = np.array(traj[k])
        traj['dones'][-1] = 1

    # print(trajectories)

    return trajectories

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def write_summary(writer, info, step):
    """For pytorch. Write summary to tensorboard."""
    for key, val in info.items():
        if isinstance(val, (int, float, np.int32, np.int64, np.float32, np.float64)):
            writer.add_scalar(key, val, step)

if __name__ == '__main__':
    # read_dst(f'data/sepsis_demo_preprocessed.csv')
    a = np.array([[1, 2, 3], [3, 2, 1], [9, 3, 10]])
    b = np.array([2, 1, 2])
    print(get_accuracy(a, b))