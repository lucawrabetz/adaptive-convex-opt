\ Model OA
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  eta + 0 x_0_0 + 0 x_0_1 + 0 x_1_0 + 0 x_1_1
Subject To
 initial_constr: eta >= 0
 R1: z_0_0 + z_0_1 = 1
 R2: z_1_0 + z_1_1 = 1
 R3: z_2_0 + z_2_1 = 1
Bounds
 x_0_0 <= 2
 x_0_1 <= 2
 x_1_0 <= 2
 x_1_1 <= 2
Binaries
 z_0_0 z_1_0 z_2_0 z_0_1 z_1_1 z_2_1
End
