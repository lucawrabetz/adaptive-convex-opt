\ Model OA
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  eta
Subject To
 initial_constr: eta >= 0
 R1: z_0_0 + z_0_1 = 1
 R2: z_1_0 + z_1_1 = 1
 R3: z_2_0 + z_2_1 = 1
 R4: z_0_0 + z_1_0 + z_2_0 >= 1
 R5: z_0_1 + z_1_1 + z_2_1 >= 1
 R6: eta - 2 x_0_0 - 9 z_0_0 >= -10
 R7: eta - 2 x_1_0 - 9 z_0_1 >= -10
 R8: eta + 2 x_0_0 - 9 z_0_0 >= -10
 R9: eta + 2 x_1_0 - 9 z_0_1 >= -10
 R10: eta - 2 x_0_1 - 9 z_0_0 >= -10
 R11: eta - 2 x_1_1 - 9 z_0_1 >= -10
 R12: eta + 2 x_0_1 - 9 z_0_0 >= -10
 R13: eta + 2 x_1_1 - 9 z_0_1 >= -10
 R14: eta - 2 x_0_0 - 9 z_1_0 >= -10
 R15: eta - 2 x_1_0 - 9 z_1_1 >= -10
 R16: eta + 2 x_0_0 - 9 z_1_0 >= -10
 R17: eta + 2 x_1_0 - 9 z_1_1 >= -10
 R18: eta - 2 x_0_1 - 9 z_1_0 >= -12
 R19: eta - 2 x_1_1 - 9 z_1_1 >= -12
 R20: eta + 2 x_0_1 - 9 z_1_0 >= -8
 R21: eta + 2 x_1_1 - 9 z_1_1 >= -8
 R22: eta - 2 x_0_0 - 9 z_2_0 >= -10
 R23: eta - 2 x_1_0 - 9 z_2_1 >= -10
 R24: eta + 2 x_0_0 - 9 z_2_0 >= -10
 R25: eta + 2 x_1_0 - 9 z_2_1 >= -10
 R26: eta - 2 x_0_1 - 9 z_2_0 >= -16
 R27: eta - 2 x_1_1 - 9 z_2_1 >= -16
 R28: eta + 2 x_0_1 - 9 z_2_0 >= -4
 R29: eta + 2 x_1_1 - 9 z_2_1 >= -4
Bounds
 x_0_0 = 0
 x_0_1 <= 3
 x_1_0 = 0
 x_1_1 <= 3
Binaries
 z_0_0 z_1_0 z_2_0 z_0_1 z_1_1 z_2_1
End
