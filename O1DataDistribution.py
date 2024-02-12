
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Class 1
Center1 = np.array([4, 4])
StdDev1 = np.array([0.8, 0.8])
NoEx1 = 100
Data = np.zeros((NoEx1, len(Center1)))

for j in range(NoEx1):
    for i in range(len(Center1)):
        Data[j, i] = Center1[i] + StdDev1[i] * np.random.randn()

plt.figure(1)
plt.plot(Data[:, 0], Data[:, 1], 'b*')
DataC1M1 = Data.copy()

Center2 = np.array([6, 6])
StdDev2 = np.array([0.8, 0.8])
NoEx2 = 100
Data = np.zeros((NoEx2, len(Center2)))

for j in range(NoEx2):
    for i in range(len(Center2)):
        Data[j, i] = Center2[i] + StdDev2[i] * np.random.randn()

plt.figure(1)
plt.plot(Data[:, 0], Data[:, 1], 'ko')
plt.title('200 initial samples - no concept drift')
plt.axis([0, 10, 0, 10])
plt.xlabel('x1')
plt.ylabel('x2')
DataC2M1 = Data.copy()

plt.show()

# Initial location of the twin Gaussians
x = np.arange(0, 10.1, 0.1)
y = np.arange(0, 10.1, 0.1)
C1 = np.zeros((len(x), len(y)))
C2 = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        C1[i, j] = np.exp(-((x[i] - Center1[0])**2 / (2 * StdDev1[0]**2) + (y[j] - Center1[1])**2 / (2 * StdDev1[1]**2)))
        C2[i, j] = np.exp(-((x[i] - Center2[0])**2 / (2 * StdDev2[0]**2) + (y[j] - Center2[1])**2 / (2 * StdDev2[1]**2)))
        
fig = go.Figure(data=[go.Surface(z=C1, x=x, y=y, colorscale='Viridis', name='Gaussian 1')])

# Add the second Gaussian to the plot
fig.add_trace(go.Surface(z=C2, x=x, y=y, colorscale='Viridis', name='Gaussian 2'))

# Set layout
fig.update_layout(scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='C'),
                    ),
                    title='3D Gaussian Plot with Two Gaussians',
                    )

fig.show()

# 90 degree anticlockwise rotation around (5,5)
# Initial position of Class 1
theta1 = 225 * np.pi / 180
Dtheta1 = 0.45 * np.pi / 180
StdDev1 = np.array([0.8, 0.8])
NoEx1 = 200
Data1 = np.zeros((NoEx1, len(Center1)))
cc1 = np.zeros((NoEx1, len(Center1)))

for j in range(NoEx1):
    Center1 = [5 + np.sqrt(2) * np.cos(theta1 + Dtheta1), 5 + np.sqrt(2) * np.sin(theta1 + Dtheta1)]
    theta1 = theta1 + Dtheta1
    for i in range(len(Center1)):
        Data1[j, i] = Center1[i] + StdDev1[i] * np.random.randn()
        cc1[j, i] = Center1[i]

plt.figure(3)
plt.plot(Data1[:, 0], Data1[:, 1], 'b*')
plt.plot(cc1[:, 0], cc1[:, 1], 'r*')
DataC1M2 = Data1.copy()

# 90 degree anticlockwise rotation around (5,5)
# Initial position of Class 2
theta2 = 45 * np.pi / 180
Dtheta2 = 0.45 * np.pi / 180
StdDev2 = np.array([0.8, 0.8])
NoEx2 = 200
Data2 = np.zeros((NoEx2, len(Center2)))
cc2 = np.zeros((NoEx2, len(Center2)))

for j in range(NoEx2):
    Center2 = [5 + np.sqrt(2) * np.cos(theta2 + Dtheta2), 5 + np.sqrt(2) * np.sin(theta2 + Dtheta2)]
    theta2 = theta2 + Dtheta2
    for i in range(len(Center2)):
        Data2[j, i] = Center2[i] + StdDev2[i] * np.random.randn()
        cc2[j, i] = Center2[i]

plt.figure(3)
plt.axis([0, 10, 0, 10])
plt.plot(Data2[:, 0], Data2[:, 1], 'ko')
plt.plot(cc2[:, 0], cc2[:, 1], 'ro')
plt.title('Samples during concept drift')
plt.xlabel('x1')
plt.ylabel('x2')
DataC2M2 = Data2.copy()

plt.show()
# Final location of the twin Gaussians
x = np.arange(0, 10.1, 0.1)
y = np.arange(0, 10.1, 0.1)
C1 = np.zeros((len(x), len(y)))
C2 = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        C1[i, j] = np.exp(-((x[i] - Center1[0])**2 / (2 * StdDev1[0]**2) + (y[j] - Center1[1])**2 / (2 * StdDev1[1]**2)))
        C2[i, j] = np.exp(-((x[i] - Center2[0])**2 / (2 * StdDev2[0]**2) + (y[j] - Center2[1])**2 / (2 * StdDev2[1]**2)))
fig = go.Figure(data=[go.Surface(z=C1, x=x, y=y, colorscale='Viridis', name='Gaussian 1')])

# Add the second Gaussian to the plot
fig.add_trace(go.Surface(z=C2, x=x, y=y, colorscale='Viridis', name='Gaussian 2'))

# Set layout
fig.update_layout(scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='C'),
                    ),
                    title='3D Gaussian Plot with Two Gaussians',
                    )

fig.show()
# Adding new column representing class label: [x C]
# 1st moment
DataC1M1 = np.hstack((DataC1M1, np.zeros((DataC1M1.shape[0], 1))))
DataC2M1 = np.hstack((DataC2M1, np.ones((DataC2M1.shape[0], 1))))

# 2nd moment
DataC1M2 = np.hstack((DataC1M2, np.zeros((DataC1M2.shape[0], 1))))
DataC2M2 = np.hstack((DataC2M2, np.ones((DataC2M2.shape[0], 1))))

# Shuffle the data
# 1st moment
DataM1 = np.vstack((DataC1M1, DataC2M1))
for _ in range(1000):
    j = np.random.randint(0, DataM1.shape[0])
    k = np.random.randint(0, DataM1.shape[0])
    DataM1[[j, k], :] = DataM1[[k, j], :]

# 2nd moment
DataM2 = np.zeros((0, DataC1M2.shape[1]))
for i in range(DataC1M2.shape[0]):
    if np.round(np.random.rand()) == 0:
        DataM2 = np.vstack((DataM2, DataC1M2[i, :]))
    else:
        DataM2 = np.vstack((DataM2, DataC2M2[i, :]))

Data = np.vstack((DataM1, DataM2))
# Save the data
np.save('DataStream.npy', Data)