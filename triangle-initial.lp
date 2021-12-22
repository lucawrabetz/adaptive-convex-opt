\ Model OA
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  eta
Subject To
 initial_constr: eta >= 0
 R1: z_0_0 + z_0_1 = 1
 R2: z_1_0 + z_1_1 = 1
 R3: z_2_0 + z_2_1 = 1
 R4: eta - 2 x_0_0 - 2 x_0_1 - 10 z_0_0 >= -12
 R5: eta - 2 x_1_0 - 2 x_1_1 - 10 z_0_1 >= -12
 R6: eta - 2 x_0_0 - 2 x_0_1 - 10 z_1_0 >= -14
 R7: eta - 2 x_1_0 - 2 x_1_1 - 10 z_1_1 >= -14
 R8: eta - 2 x_0_0 - 2 x_0_1 - 10 z_2_0 >= -16
 R9: eta - 2 x_1_0 - 2 x_1_1 - 10 z_2_1 >= -16
Bounds
Binaries
 z_0_0 z_1_0 z_2_0 z_0_1 z_1_1 z_2_1
End
