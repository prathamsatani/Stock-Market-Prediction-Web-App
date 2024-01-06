import pandas as pd

ma1 = [x for x in range(10)]
ma2 = [x*2 for x in range(10)]
ma3 = [x*3 for x in range(10)]

ma = pd.DataFrame([ma1, ma2, ma3])
print(ma)