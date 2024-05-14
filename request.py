import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Dst_Port': 40,'Protocol':6,'Flow_Duration': 30, 'Subflow_Bwd_Byts':1.0,'Idle_Std':0, 'Subflow_Fwd_Pkts':10,'Bwd_Pkts':0,'Flow_Byts':0})

print(r.json())