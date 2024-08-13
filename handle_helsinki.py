from helpful_scripts.new_dataset import NewDataset

path = '/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/default_scenario_MovementNs2Report.txt'

helsinki = NewDataset('helsinki', path)
helsinki.export_raw_data()