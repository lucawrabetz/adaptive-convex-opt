\ Model OA
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  eta + 0 x_0_0 + 0 x_0_1 + 0 x_1_0 + 0 x_1_1
Subject To
 initial_constr: eta >= 0
 R1: z_0_0 + z_0_1 = 1
 R2: z_1_0 + z_1_1 = 1
 R3: z_2_0 + z_2_1 = 1
 R4: z_0_0 + z_1_0 + z_2_0 >= 1
 R5: z_0_1 + z_1_1 + z_2_1 >= 1
 initial_linear_0_0.01_0_0: eta - 9 z_0_0 >= -9
 initial_linear_0_0.01_1_0: eta - 9 z_0_1 >= -9
 initial_linear_0_0.01_0_1: eta - 9 z_0_0 >= -9
 initial_linear_0_0.01_1_1: eta - 9 z_0_1 >= -9
 initial_linear_0_0.01_0_2: eta - 9 z_0_0 >= -9
 initial_linear_0_0.01_1_2: eta - 9 z_0_1 >= -9
 initial_linear_0_0.01_0_3: eta - 9 z_0_0 >= -9
 initial_linear_0_0.01_1_3: eta - 9 z_0_1 >= -9
 initial_linear_1_0.01_0_0: eta - 9 z_1_0 >= -9
 initial_linear_1_0.01_1_0: eta - 9 z_1_1 >= -9
 initial_linear_1_0.01_0_1: eta - 9 z_1_0 >= -9
 initial_linear_1_0.01_1_1: eta - 9 z_1_1 >= -9
 initial_linear_1_0.01_0_2: eta - 9 z_1_0 >= -9
 initial_linear_1_0.01_1_2: eta - 9 z_1_1 >= -9
 initial_linear_1_0.01_0_3: eta - 9 z_1_0 >= -9
 initial_linear_1_0.01_1_3: eta - 9 z_1_1 >= -9
 initial_linear_2_0.01_0_0: eta - 9 z_2_0 >= -9
 initial_linear_2_0.01_1_0: eta - 9 z_2_1 >= -9
 initial_linear_2_0.01_0_1: eta - 9 z_2_0 >= -9
 initial_linear_2_0.01_1_1: eta - 9 z_2_1 >= -9
 initial_linear_2_0.01_0_2: eta - 9 z_2_0 >= -9
 initial_linear_2_0.01_1_2: eta - 9 z_2_1 >= -9
 initial_linear_2_0.01_0_3: eta - 9 z_2_0 >= -9
 initial_linear_2_0.01_1_3: eta - 9 z_2_1 >= -9
 initial_linear_0_0.05_0_0: eta - 9 z_0_0 >= -9
 initial_linear_0_0.05_1_0: eta - 9 z_0_1 >= -9
 initial_linear_0_0.05_0_1: eta - 9 z_0_0 >= -9
 initial_linear_0_0.05_1_1: eta - 9 z_0_1 >= -9
 initial_linear_0_0.05_0_2: eta - 9 z_0_0 >= -9
 initial_linear_0_0.05_1_2: eta - 9 z_0_1 >= -9
 initial_linear_0_0.05_0_3: eta - 9 z_0_0 >= -9
 initial_linear_0_0.05_1_3: eta - 9 z_0_1 >= -9
 initial_linear_1_0.05_0_0: eta - 9 z_1_0 >= -9
 initial_linear_1_0.05_1_0: eta - 9 z_1_1 >= -9
 initial_linear_1_0.05_0_1: eta - 9 z_1_0 >= -9
 initial_linear_1_0.05_1_1: eta - 9 z_1_1 >= -9
 initial_linear_1_0.05_0_2: eta - 9 z_1_0 >= -9
 initial_linear_1_0.05_1_2: eta - 9 z_1_1 >= -9
 initial_linear_1_0.05_0_3: eta - 9 z_1_0 >= -9
 initial_linear_1_0.05_1_3: eta - 9 z_1_1 >= -9
 initial_linear_2_0.05_0_0: eta - 9 z_2_0 >= -9
 initial_linear_2_0.05_1_0: eta - 9 z_2_1 >= -9
 initial_linear_2_0.05_0_1: eta - 9 z_2_0 >= -9
 initial_linear_2_0.05_1_1: eta - 9 z_2_1 >= -9
 initial_linear_2_0.05_0_2: eta - 9 z_2_0 >= -9
 initial_linear_2_0.05_1_2: eta - 9 z_2_1 >= -9
 initial_linear_2_0.05_0_3: eta - 9 z_2_0 >= -9
 initial_linear_2_0.05_1_3: eta - 9 z_2_1 >= -9
 initial_linear_0_0.1_0_0: eta - 9 z_0_0 >= -9
 initial_linear_0_0.1_1_0: eta - 9 z_0_1 >= -9
 initial_linear_0_0.1_0_1: eta - 9 z_0_0 >= -9
 initial_linear_0_0.1_1_1: eta - 9 z_0_1 >= -9
 initial_linear_0_0.1_0_2: eta - 9 z_0_0 >= -9
 initial_linear_0_0.1_1_2: eta - 9 z_0_1 >= -9
 initial_linear_0_0.1_0_3: eta - 9 z_0_0 >= -9
 initial_linear_0_0.1_1_3: eta - 9 z_0_1 >= -9
 initial_linear_1_0.1_0_0: eta - 9 z_1_0 >= -9
 initial_linear_1_0.1_1_0: eta - 9 z_1_1 >= -9
 initial_linear_1_0.1_0_1: eta - 9 z_1_0 >= -9
 initial_linear_1_0.1_1_1: eta - 9 z_1_1 >= -9
 initial_linear_1_0.1_0_2: eta - 9 z_1_0 >= -9
 initial_linear_1_0.1_1_2: eta - 9 z_1_1 >= -9
 initial_linear_1_0.1_0_3: eta - 9 z_1_0 >= -9
 initial_linear_1_0.1_1_3: eta - 9 z_1_1 >= -9
 initial_linear_2_0.1_0_0: eta - 9 z_2_0 >= -9
 initial_linear_2_0.1_1_0: eta - 9 z_2_1 >= -9
 initial_linear_2_0.1_0_1: eta - 9 z_2_0 >= -9
 initial_linear_2_0.1_1_1: eta - 9 z_2_1 >= -9
 initial_linear_2_0.1_0_2: eta - 9 z_2_0 >= -9
 initial_linear_2_0.1_1_2: eta - 9 z_2_1 >= -9
 initial_linear_2_0.1_0_3: eta - 9 z_2_0 >= -9
 initial_linear_2_0.1_1_3: eta - 9 z_2_1 >= -9
Bounds
 x_0_0 <= 3
 x_0_1 <= 3
 x_1_0 <= 3
 x_1_1 <= 3
Binaries
 z_0_0 z_1_0 z_2_0 z_0_1 z_1_1 z_2_1
End
