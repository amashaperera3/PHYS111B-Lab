#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[39]:


#Semiconductor Dimensions
width = 10.0/100
length = 10.0/100
thickness = 1.25/1000
cross_area = width * thickness
I = 100e-6
q = 1.602e-19


# ## 2000 G B-field ##

# In[115]:


# Load the 2000G raw data file
df_2000G = pd.read_csv("2000G_100uA.csv", sep="\t", engine="python")

# Convert required columns to numeric
cols_to_convert = ["Temperature (K)", "Voltage AC", "Voltage BD", "B-Field (Gauss)"]
for col in cols_to_convert:
    df_2000G[col] = pd.to_numeric(df_2000G[col], errors="coerce")

# Drop rows with missing values in any of those columns
df_2000G = df_2000G.dropna(subset=cols_to_convert)

# Split the data into three B-field ranges
df2000_B_neg = df_2000G[df_2000G["B-Field (Gauss)"] < -1000].copy()
df2000_B_zero = df_2000G[(df_2000G["B-Field (Gauss)"] >= -1000) & (df_2000G["B-Field (Gauss)"] <= 1000)].copy()
df2000_B_pos = df_2000G[df_2000G["B-Field (Gauss)"] > 1000].copy()

for df in [df2000_B_neg, df2000_B_zero, df2000_B_pos]:
    df["1/T (1/K)"] = 1 / pd.to_numeric(df["Temperature (K)"])
    df["B (T)"] = df["B-Field (Gauss)"] / 1e4

df2_B_neg = df2000_B_neg[["Temperature (K)", "1/T (1/K)", "Voltage AC", "Voltage BD", "B (T)"]].to_numpy()
df2_B_zero = df2000_B_zero[["Temperature (K)", "1/T (1/K)", "Voltage AC", "Voltage BD", "B (T)"]].to_numpy()
df2_B_pos = df2000_B_pos[["Temperature (K)", "1/T (1/K)", "Voltage AC", "Voltage BD", "B (T)"]].to_numpy()


# In[71]:


#ROOM TEMPERATURE CALCULATION
df_2000G['Temperature_Diff'] = np.abs(df_2000G['Temperature (K)'] - 300)
room_temp_row = df_2000G.loc[df_2000G['Temperature_Diff'].idxmin()]

T = room_temp_row['Temperature (K)']
Res_avg = (room_temp_row['Voltage AC'] / room_temp_row['sample I BD'] +
                         room_temp_row['Voltage BD'] / room_temp_row['sample I AC']) / 2
Res = np.abs(Res_avg)
V_hall = (room_temp_row['Voltage BD'] - room_temp_row['Voltage -BD']) / 2        # Hall voltage
B_gauss = room_temp_row['B-Field (Gauss)']
B_tesla = B_gauss * 1e-4  # In Tesla

# Resistivity
rho = Res*0.01 * (cross_area / length)

# Hall Coefficient
RH = (V_hall * thickness) / (I * B_tesla)

# Magnetic field where E_Hall = E_resistive
df_RT = df_2000G[(df_2000G["Temperature (K)"] >= 295) & (df_2000G["Temperature (K)"] <= 305)].copy()
df_RT["|V_H - V_R|"] = np.abs(df_RT["Voltage BD"] - df_RT["Voltage AC"])

# Step 3: Find row with minimum difference
best_match_idx = df_RT["|V_H - V_R|"].idxmin()
match = df_RT.loc[best_match_idx]


# Output
print("Since the length and width of the semiconductor are approximately equal, we can simply look at the B-field when the hall and resistive voltages are approximately equal.")
print(f"Room Temperature (K): {T}")
print(f"Resistance (Ohm): {Res:.3f}")
print(f"Resistivity (Ohm·m): {rho:.6f}")
print(f"Hall Coefficient RH (m^3/C): {RH:.6f}")
print(f"B-field (Gauss) where V_H ≈ V_R: {match['B-Field (Gauss)']:.2f} G")


# In[109]:


neg_temp = df2_B_neg[:, 0]
neginv_temp = df2_B_neg[:, 1]
negvolt_ac = df2_B_neg[:, 2]
negvolt_bd = df2_B_neg[:, 3]
negb_field = df2_B_neg[:, 4]

pos_temp = df2_B_pos[:, 0]
posinv_temp = df2_B_pos[:, 1]
posvolt_ac = df2_B_pos[:, 2]
posvolt_bd = df2_B_pos[:, 3]
posb_field = df2_B_pos[:, 4]

zero_temp = df2_B_zero[:, 0]
zeroinv_temp = df2_B_zero[:, 1]
zerovolt_ac = df2_B_zero[:, 2]
zerovolt_bd = df2_B_zero[:, 3]
zerob_field = df2_B_zero[:, 4]


# In[110]:


# Compute physical quantities - Negative
neg_resis = np.abs((negvolt_ac / I) * 0.01 * cross_area / length)
neg_conduc = 1 / neg_resis
neg_RH = (negvolt_bd * thickness) / (I * negb_field)
neg_concentration = 1 / (np.abs(neg_RH) * q)
neg_mobility = np.abs(neg_RH) * neg_conduc
neg_RH·σ = neg_RH * neg_conduc
neg_semi_type = np.where(neg_RH > 0, "p-type", "n-type")

# Compute physical quantities - Positive
pos_resis = np.abs((posvolt_ac*0.01 * cross_area) / (I * length))
pos_conduc = 1 / pos_resis
pos_RH = (posvolt_bd * thickness) / (I * posb_field)
pos_concentration = 1 / (np.abs(pos_RH) * q)
pos_mobility = np.abs(pos_RH) * pos_conduc
pos_RH·σ = pos_RH * pos_conduc
pos_semi_type = np.where(pos_RH > 0, "p-type", "n-type")

# Compute physical quantities - Zero
zero_resis = np.abs((zerovolt_ac*0.01 * cross_area) / (I * length))
zero_conduc = 1 / zero_resis
zero_RH = (zerovolt_bd * thickness) / (I * zerob_field)
zero_concentration = 1 / (np.abs(zero_RH) * q)
zero_mobility = np.abs(zero_RH) * zero_conduc
zero_RH·σ = zero_RH * zero_conduc
zero_semi_type = np.where(zero_RH > 0, "p-type", "n-type") 


# Electron/Hole separation
neg_n_holes = np.where(neg_RH > 0, neg_concentration, np.nan)
neg_n_elec = np.where(neg_RH < 0, neg_concentration, np.nan)
neg_mu_holes = np.where(neg_RH > 0, neg_mobility, np.nan)
neg_mu_elec = np.where(neg_RH < 0, neg_mobility, np.nan)

pos_n_holes = np.where(neg_RH > 0, neg_concentration, np.nan)
pos_n_elec = np.where(neg_RH < 0, neg_concentration, np.nan)
pos_mu_holes = np.where(neg_RH > 0, neg_mobility, np.nan)
pos_mu_elec = np.where(neg_RH < 0, neg_mobility, np.nan)

zero_n_holes = np.where(neg_RH > 0, neg_concentration, np.nan)
zero_n_elec = np.where(neg_RH < 0, neg_concentration, np.nan)
zero_mu_holes = np.where(neg_RH > 0, neg_mobility, np.nan)
zero_mu_elec = np.where(neg_RH < 0, neg_mobility, np.nan)


# neg_resis , 
# neg_conduc , 
# neg_RH , 
# neg_concentration , 
# neg_mobility , 
# neg_RH·σ , 
# neg_semi_type
# 
# zero_resis , 
# zero_conduc , 
# zero_RH , 
# zero_concentration , 
# zero_mobility , 
# zero_RH·σ , 
# zero_semi_type
# 
# pos_resis , 
# pos_conduc , 
# pos_RH , 
# pos_concentration , 
# pos_mobility , 
# pos_RH·σ , 
# pos_semi_type

# In[107]:


# RESISTIVITY VS INVERSE TEMPERATURE
def plot_three_b_field_resis():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # B < -100 G
    axs[0].plot(neginv_temp, neg_resis, '-o')
    axs[0].set_title("B < -100 G")
    axs[0].set_xlabel("1/T (1/K)")
    axs[0].set_ylabel("Resistivity [Ω·m]")
    #axs[0].set_ylim(-0.002, 0.008)
    #axs[0].legend()
    axs[0].grid(True)

    # -100 G ≤ B ≤ 100 G
    axs[1].plot(zeroinv_temp, zero_resis, '-o')
    axs[1].set_title("-100 G ≤ B ≤ 100 G")
    axs[1].set_xlabel("1/T (1/K)")
    #axs[1].set_ylim(-0.002, 0.004)
    #axs[1].legend()
    axs[1].grid(True)

    # B > 100 G
    axs[2].plot(posinv_temp, pos_resis, '-o')
    axs[2].set_title("B > 100 G")
    axs[2].set_xlabel("1/T (1/K)")
    #axs[2].set_ylim(-0.002, 0.014)
    #axs[2].legend()
    axs[2].grid(True)
    
    plt.suptitle('Resistivity vs Inverse Temperature \n B-field: 2000G, Current: 100uA', fontsize=18)

    plt.tight_layout()
    plt.show()
    
plot_three_b_field_resis()


# ["RH (m^3/C)"]
# ["Carrier Concentration (1/m^3)"]
# ["Mobility (m^2/Vs)"]

# In[102]:


#CONDUCTIVITY VS INVERSE TEMPERATURE
def plot_three_b_field_conduc():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # B < -100 G
    axs[0].plot(neginv_temp, neg_conduc, '-o')
    axs[0].set_title("B < -100 G")
    axs[0].set_xlabel("1/T (1/K)")
    axs[0].set_ylabel("Conductivity (S/m)")
    #axs[0].legend()
    axs[0].grid(True)

    # -100 G ≤ B ≤ 100 G
    axs[1].plot(zeroinv_temp, zero_conduc, '-o')
    axs[1].set_title("-100 G ≤ B ≤ 100 G")
    axs[1].set_xlabel("1/T (1/K)")
    #axs[1].legend()
    axs[1].grid(True)

    # B > 100 G
    axs[2].plot(posinv_temp, pos_conduc, '-o')
    axs[2].set_title("B > 100 G")
    axs[2].set_xlabel("1/T (1/K)")
    #axs[2].legend()
    axs[2].grid(True)
    
    # Ensure y-ticks and labels on all plots
    for ax in axs:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
        ax.tick_params(labelleft=True)
    
    plt.suptitle('Conductivity vs Inverse Temperature \n B-field: 2000G, Current: 100uA', fontsize=18)

    plt.tight_layout()
    plt.show()
    
plot_three_b_field_conduc()


# In[106]:


# HALL COEFF VS INVERSE TEMPERATURE
def plot_three_b_field_hall():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # B < -100 G
    axs[0].plot(neginv_temp, neg_RH, '-o')
    axs[0].set_title("B < -100 G")
    axs[0].set_xlabel("1/T (1/K)")
    axs[0].set_ylabel("RH (m^3/C)")
    #axs[0].set_ylim(-10, 5)
    #axs[0].legend()
    axs[0].grid(True)

    # -100 G ≤ B ≤ 100 G
    axs[1].plot(zeroinv_temp, zero_RH, '-o')
    axs[1].set_title("-100 G ≤ B ≤ 100 G")
    axs[1].set_xlabel("1/T (1/K)")
    #axs[0].set_ylim(-90, 10)
    #axs[1].legend()
    axs[1].grid(True)

    # B > 100 G
    axs[2].plot(posinv_temp, pos_RH, '-o')
    axs[2].set_title("B > 100 G")
    axs[2].set_xlabel("1/T (1/K)")
    #axs[0].set_ylim(-10, 10)
    #axs[2].legend()
    axs[2].grid(True)
    
    plt.suptitle('Hall Coefficient vs Inverse Temperature \n B-field: 2000G, Current: 100uA', fontsize=18)

    plt.tight_layout()
    plt.show()
    
plot_three_b_field_hall()


# In[117]:


# CONCENTRATION VS TEMPERATURE
def plot_three_b_field_concen():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # B < -100 G
    axs[0].plot(neg_temp, np.log(neg_concentration), '-o')
    axs[0].set_title("B < -100 G")
    axs[0].set_xlabel("1/T (1/K)")
    axs[0].set_ylabel("Carrier Concentration (1/m^3)")
    #axs[0].set_ylim(-10, 5)
    #axs[0].legend()
    axs[0].grid(True)

    # -100 G ≤ B ≤ 100 G
    axs[1].plot(zero_temp, np.log(zero_concentration), '-o')
    axs[1].set_title("-100 G ≤ B ≤ 100 G")
    axs[1].set_xlabel("1/T (1/K)")
    #axs[0].set_ylim(-90, 10)
    #axs[1].legend()
    axs[1].grid(True)

    # B > 100 G
    axs[2].plot(pos_temp, np.log(pos_concentration), '-o')
    axs[2].set_title("B > 100 G")
    axs[2].set_xlabel("1/T (1/K)")
    #axs[0].set_ylim(-10, 10)
    #axs[2].legend()
    axs[2].grid(True)
    
    plt.suptitle('Concentration vs Inverse Temperature \n B-field: 2000G, Current: 100uA', fontsize=18)

    plt.tight_layout()
    plt.show()
    
plot_three_b_field_concen()


# In[113]:


# MOBILITY VS TEMPERATURE
def plot_three_b_field_mobil():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # B < -100 G
    axs[0].plot(neg_temp, neg_mobility, '-o')
    axs[0].set_title("B < -100 G")
    axs[0].set_xlabel("1/T (1/K)")
    axs[0].set_ylabel("Mobility (m^2/Vs)")
    #axs[0].set_ylim(-10, 5)
    #axs[0].legend()
    axs[0].grid(True)

    # -100 G ≤ B ≤ 100 G
    axs[1].plot(zero_temp, zero_mobility, '-o')
    axs[1].set_title("-100 G ≤ B ≤ 100 G")
    axs[1].set_xlabel("1/T (1/K)")
    #axs[0].set_ylim(-90, 10)
    #axs[1].legend()
    axs[1].grid(True)

    # B > 100 G
    axs[2].plot(pos_temp, pos_mobility, '-o')
    axs[2].set_title("B > 100 G")
    axs[2].set_xlabel("1/T (1/K)")
    #axs[0].set_ylim(-10, 10)
    #axs[2].legend()
    axs[2].grid(True)
    
    plt.suptitle('Carrier Mobility vs Inverse Temperature \n B-field: 2000G, Current: 100uA', fontsize=18)

    plt.tight_layout()
    plt.show()
    
plot_three_b_field_mobil()


# In[ ]:




