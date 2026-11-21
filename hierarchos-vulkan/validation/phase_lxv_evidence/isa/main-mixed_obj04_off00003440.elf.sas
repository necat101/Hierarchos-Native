
:	file format elf64-amdgpu

Disassembly of section .text:
_amdgpu_cs_main:
	s_getpc_b64 s[12:13]                                       // 000000000000: BE8C4700 
	s_mov_b32 s0, s9                                           // 000000000004: BE800009 
	s_mov_b32 s9, s13                                          // 000000000008: BE89000D 
	v_mbcnt_lo_u32_b32 v19, -1, 0                              // 00000000000C: D71F0013 000100C1 
	s_load_b128 s[20:23], s[8:9], 0x20                         // 000000000014: F4080504 F8000020 
	v_mov_b32_e32 v15, 0                                       // 00000000001C: 7E1E0280 
	s_lshl_b32 s11, s10, 5                                     // 000000000020: 840B850A 
	s_lshl_b32 s10, s0, 5                                      // 000000000024: 840A8500 
	v_ashrrev_i32_e32 v21, 31, v19                             // 000000000028: 342A269F 
	v_and_b32_e32 v20, 15, v19                                 // 00000000002C: 3628268F 
	v_mov_b32_e32 v14, v15                                     // 000000000030: 7E1C030F 
	v_mov_b32_e32 v13, v15                                     // 000000000034: 7E1A030F 
	v_mov_b32_e32 v12, v15                                     // 000000000038: 7E18030F 
	v_mov_b32_e32 v11, v15                                     // 00000000003C: 7E16030F 
	v_mov_b32_e32 v10, v15                                     // 000000000040: 7E14030F 
	v_mov_b32_e32 v9, v15                                      // 000000000044: 7E12030F 
	v_mov_b32_e32 v8, v15                                      // 000000000048: 7E10030F 
	v_mov_b32_e32 v7, v15                                      // 00000000004C: 7E0E030F 
	v_mov_b32_e32 v6, v15                                      // 000000000050: 7E0C030F 
	v_mov_b32_e32 v5, v15                                      // 000000000054: 7E0A030F 
	v_mov_b32_e32 v4, v15                                      // 000000000058: 7E08030F 
	v_mov_b32_e32 v3, v15                                      // 00000000005C: 7E06030F 
	v_mov_b32_e32 v2, v15                                      // 000000000060: 7E04030F 
	v_mov_b32_e32 v1, v15                                      // 000000000064: 7E02030F 
	v_mov_b32_e32 v0, v15                                      // 000000000068: 7E00030F 
	s_cmp_eq_u32 s3, 0                                         // 00000000006C: BF068003 
	s_mov_b32 s24, 0                                           // 000000000070: BE980080 
	s_cbranch_scc1 226                                         // 000000000074: BFA200E2  <_amdgpu_cs_main+0x400>
	v_lshrrev_b32_e32 v0, 28, v21                              // 000000000078: 32002A9C 
	s_load_b256 s[12:19], s[8:9], null                         // 00000000007C: F40C0304 F8000000 
	v_mov_b32_e32 v8, 0                                        // 000000000084: 7E100280 
	s_bitcmp1_b32 s7, 0                                        // 000000000088: BF0D8007 
	v_cmp_gt_u32_e64 s0, 0x200, v19                            // 00000000008C: D44C0000 000226FF 00000200 
	v_add_nc_u32_e32 v0, v19, v0                               // 000000000098: 4A000113 
	v_or_b32_e32 v22, s10, v20                                 // 00000000009C: 382C280A 
	v_mov_b32_e32 v9, v8                                       // 0000000000A0: 7E120308 
	v_mov_b32_e32 v10, v8                                      // 0000000000A4: 7E140308 
	v_mov_b32_e32 v12, v8                                      // 0000000000A8: 7E180308 
	v_dual_mov_b32 v11, v8 :: v_dual_and_b32 v0, -16, v0       // 0000000000AC: CA240108 0B0000D0 
	v_mov_b32_e32 v14, v8                                      // 0000000000B4: 7E1C0308 
	v_mov_b32_e32 v15, v8                                      // 0000000000B8: 7E1E0308 
	v_mov_b32_e32 v2, v8                                       // 0000000000BC: 7E040308 
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000000C0: BF870004 
	v_sub_nc_u32_e32 v1, v19, v0                               // 0000000000C4: 4C020113 
	v_mov_b32_e32 v13, v8                                      // 0000000000C8: 7E1A0308 
	v_mov_b32_e32 v0, v8                                       // 0000000000CC: 7E000308 
	v_mov_b32_e32 v3, v8                                       // 0000000000D0: 7E060308 
	v_mov_b32_e32 v4, v8                                       // 0000000000D4: 7E080308 
	v_lshl_add_u32 v23, v1, 5, 0x400                           // 0000000000D8: D6460017 03FD0B01 00000400 
	v_dual_mov_b32 v1, v8 :: v_dual_lshlrev_b32 v24, 1, v1     // 0000000000E4: CA220108 01180281 
	v_mov_b32_e32 v5, v8                                       // 0000000000EC: 7E0A0308 
	v_mov_b32_e32 v6, v8                                       // 0000000000F0: 7E0C0308 
	v_mov_b32_e32 v7, v8                                       // 0000000000F4: 7E0E0308 
	s_cselect_b32 s25, -1, 0                                   // 0000000000F8: 981980C1 
	s_bitcmp0_b32 s7, 1                                        // 0000000000FC: BF0C8107 
	s_cselect_b32 s7, -1, 0                                    // 000000000100: 980780C1 
	s_branch 104                                               // 000000000104: BFA00068  <_amdgpu_cs_main+0x2A8>
	s_or_b32 exec_lo, exec_lo, s26                             // 000000000108: 8C7E1A7E 
	s_waitcnt lgkmcnt(0)                                       // 00000000010C: BF89FC07 
	ds_load_u16_d16 v25, v24                                   // 000000000110: DA980000 19000018 
	ds_load_u16_d16 v26, v24 offset:64                         // 000000000118: DA980040 1A000018 
	ds_load_u16_d16 v27, v24 offset:128                        // 000000000120: DA980080 1B000018 
	ds_load_u16_d16 v28, v24 offset:192                        // 000000000128: DA9800C0 1C000018 
	ds_load_u16_d16 v29, v24 offset:256                        // 000000000130: DA980100 1D000018 
	ds_load_u16_d16 v30, v24 offset:320                        // 000000000138: DA980140 1E000018 
	ds_load_u16_d16 v31, v24 offset:384                        // 000000000140: DA980180 1F000018 
	ds_load_u16_d16 v32, v24 offset:448                        // 000000000148: DA9801C0 20000018 
	ds_load_u16_d16 v33, v24 offset:512                        // 000000000150: DA980200 21000018 
	ds_load_u16_d16 v34, v24 offset:576                        // 000000000158: DA980240 22000018 
	ds_load_u16_d16 v35, v24 offset:640                        // 000000000160: DA980280 23000018 
	ds_load_u16_d16 v36, v24 offset:704                        // 000000000168: DA9802C0 24000018 
	ds_load_u16_d16 v37, v24 offset:768                        // 000000000170: DA980300 25000018 
	ds_load_u16_d16 v38, v24 offset:832                        // 000000000178: DA980340 26000018 
	ds_load_u16_d16 v39, v24 offset:896                        // 000000000180: DA980380 27000018 
	ds_load_u16_d16 v40, v24 offset:960                        // 000000000188: DA9803C0 28000018 
	ds_load_b128 v[41:44], v23                                 // 000000000190: DBFC0000 29000017 
	ds_load_b128 v[45:48], v23 offset:16                       // 000000000198: DBFC0010 2D000017 
	ds_load_b128 v[49:52], v23 offset:512                      // 0000000001A0: DBFC0200 31000017 
	ds_load_b128 v[53:56], v23 offset:528                      // 0000000001A8: DBFC0210 35000017 
	s_add_i32 s24, s24, 16                                     // 0000000001B0: 81189018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001B4: BF89FD37 
	ds_load_u16_d16_hi v25, v24 offset:32                      // 0000000001B8: DA9C0020 19000018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001C0: BF89FD37 
	ds_load_u16_d16_hi v26, v24 offset:96                      // 0000000001C4: DA9C0060 1A000018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001CC: BF89FD37 
	ds_load_u16_d16_hi v27, v24 offset:160                     // 0000000001D0: DA9C00A0 1B000018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001D8: BF89FD37 
	ds_load_u16_d16_hi v28, v24 offset:224                     // 0000000001DC: DA9C00E0 1C000018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001E4: BF89FD37 
	ds_load_u16_d16_hi v29, v24 offset:288                     // 0000000001E8: DA9C0120 1D000018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001F0: BF89FD37 
	ds_load_u16_d16_hi v30, v24 offset:352                     // 0000000001F4: DA9C0160 1E000018 
	s_waitcnt lgkmcnt(19)                                      // 0000000001FC: BF89FD37 
	ds_load_u16_d16_hi v31, v24 offset:416                     // 000000000200: DA9C01A0 1F000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000208: BF89FD37 
	ds_load_u16_d16_hi v32, v24 offset:480                     // 00000000020C: DA9C01E0 20000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000214: BF89FD37 
	ds_load_u16_d16_hi v33, v24 offset:544                     // 000000000218: DA9C0220 21000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000220: BF89FD37 
	ds_load_u16_d16_hi v34, v24 offset:608                     // 000000000224: DA9C0260 22000018 
	s_waitcnt lgkmcnt(19)                                      // 00000000022C: BF89FD37 
	ds_load_u16_d16_hi v35, v24 offset:672                     // 000000000230: DA9C02A0 23000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000238: BF89FD37 
	ds_load_u16_d16_hi v36, v24 offset:736                     // 00000000023C: DA9C02E0 24000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000244: BF89FD37 
	ds_load_u16_d16_hi v37, v24 offset:800                     // 000000000248: DA9C0320 25000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000250: BF89FD37 
	ds_load_u16_d16_hi v38, v24 offset:864                     // 000000000254: DA9C0360 26000018 
	s_waitcnt lgkmcnt(19)                                      // 00000000025C: BF89FD37 
	ds_load_u16_d16_hi v39, v24 offset:928                     // 000000000260: DA9C03A0 27000018 
	s_waitcnt lgkmcnt(19)                                      // 000000000268: BF89FD37 
	ds_load_u16_d16_hi v40, v24 offset:992                     // 00000000026C: DA9C03E0 28000018 
	s_cmp_lt_u32 s24, s3                                       // 000000000274: BF0A0318 
	s_waitcnt lgkmcnt(8)                                       // 000000000278: BF89FC87 
	v_wmma_f16_16x16x16_f16 v[8:15], v[41:48], v[25:32], v[8:15]// 00000000027C: CC424008 1C223329 
	v_wmma_f16_16x16x16_f16 v[0:7], v[49:56], v[25:32], v[0:7] // 000000000284: CC424000 1C023331 
	s_waitcnt lgkmcnt(0)                                       // 00000000028C: BF89FC07 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000290: BF870112 
	v_wmma_f16_16x16x16_f16 v[8:15], v[41:48], v[33:40], v[8:15] op_sel:[0,0,1]// 000000000294: CC426008 1C224329 
	v_wmma_f16_16x16x16_f16 v[0:7], v[49:56], v[33:40], v[0:7] op_sel:[0,0,1]// 00000000029C: CC426000 1C024331 
	s_cbranch_scc0 86                                          // 0000000002A4: BFA10056  <_amdgpu_cs_main+0x400>
	s_and_saveexec_b32 s26, s0                                 // 0000000002A8: BE9A2000 
	s_cbranch_execz 65430                                      // 0000000002AC: BFA5FF96  <_amdgpu_cs_main+0x108>
	v_or_b32_e32 v16, s24, v20                                 // 0000000002B0: 38202818 
	v_mov_b32_e32 v26, v19                                     // 0000000002B4: 7E340313 
	s_mov_b32 s27, 0                                           // 0000000002B8: BE9B0080 
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000002BC: BF870002 
	v_mul_lo_u32 v25, v16, s4                                  // 0000000002C0: D72C0019 00000910 
	v_cmp_gt_u32_e64 s8, s3, v16                               // 0000000002C8: D44C0008 00022003 
	s_branch 12                                                // 0000000002D0: BFA0000C  <_amdgpu_cs_main+0x304>
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000002D4: 8C7E097E 
	v_add_nc_u32_e32 v18, 32, v26                              // 0000000002D8: 4A2434A0 
	v_cmp_lt_u32_e32 vcc_lo, 0x1df, v26                        // 0000000002DC: 7C9234FF 000001DF 
	s_waitcnt vmcnt(0)                                         // 0000000002E4: BF8903F7 
	ds_store_b16 v17, v27                                      // 0000000002E8: D87C0000 00001B11 
	v_mov_b32_e32 v26, v18                                     // 0000000002F0: 7E340312 
	s_or_b32 s27, vcc_lo, s27                                  // 0000000002F4: 8C1B1B6A 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000002F8: BF870009 
	s_and_not1_b32 exec_lo, exec_lo, s27                       // 0000000002FC: 917E1B7E 
	s_cbranch_execz 65409                                      // 000000000300: BFA5FF81  <_amdgpu_cs_main+0x108>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000000304: BF870241 
	v_lshrrev_b32_e32 v27, 8, v26                              // 000000000308: 32363488 
	v_bfe_u32 v28, v26, 4, 4                                   // 00000000030C: D610001C 0211091A 
	v_and_b32_e32 v30, 0xff, v26                               // 000000000314: 363C34FF 000000FF 
 <unknown>
	v_lshlrev_b32_e32 v29, 4, v27                              // 000000000320: 303A3684 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000324: BF870091 
	v_or3_b32 v31, v29, s11, v28                               // 000000000328: D658001F 0470171D 
	v_cmp_gt_u32_e32 vcc_lo, s1, v31                           // 000000000330: 7C983E01 
	s_and_b32 s28, s8, vcc_lo                                  // 000000000334: 8B1C6A08 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000338: BF870009 
	s_and_saveexec_b32 s9, s28                                 // 00000000033C: BE89201C 
	s_cbranch_execz 16                                         // 000000000340: BFA50010  <_amdgpu_cs_main+0x384>
	s_and_not1_b32 vcc_lo, exec_lo, s25                        // 000000000344: 916A197E 
	s_mov_b32 s28, -1                                          // 000000000348: BE9C00C1 
	s_cbranch_vccnz 2                                          // 00000000034C: BFA40002  <_amdgpu_cs_main+0x358>
	v_add_nc_u32_e32 v17, v31, v25                             // 000000000350: 4A22331F 
	s_mov_b32 s28, 0                                           // 000000000354: BE9C0080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000358: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 00000000035C: 916A1C7E 
	s_cbranch_vccnz 3                                          // 000000000360: BFA40003  <_amdgpu_cs_main+0x370>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000364: BF870001 
	v_mad_u64_u32 v[17:18], null, v31, s4, v[16:17]            // 000000000368: D6FE7C11 0440091F 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000370: BF870001 
	v_lshlrev_b32_e32 v17, 1, v17                              // 000000000374: 30222281 
	s_waitcnt lgkmcnt(0)                                       // 000000000378: BF89FC07 
	buffer_load_u16 v18, v17, s[12:15], 0 offen                // 00000000037C: E0480000 80431211 
	s_or_b32 exec_lo, exec_lo, s9                              // 000000000384: 8C7E097E 
	v_or_b32_e32 v29, v22, v29                                 // 000000000388: 383A3B16 
	v_or_b32_e32 v28, s24, v28                                 // 00000000038C: 38383818 
	v_lshlrev_b32_e32 v17, 1, v30                              // 000000000390: 30223C81 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000000394: BF870193 
	v_cmp_gt_u32_e32 vcc_lo, s2, v29                           // 000000000398: 7C983A02 
	v_cmp_gt_u32_e64 s9, s3, v28                               // 00000000039C: D44C0009 00023803 
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000003A4: BF870003 
	v_lshl_or_b32 v17, v27, 9, v17                             // 0000000003A8: D6560011 0445131B 
 <unknown>
	s_and_b32 s28, s9, vcc_lo                                  // 0000000003B4: 8B1C6A09 
	s_waitcnt vmcnt(0)                                         // 0000000003B8: BF8903F7 
	ds_store_b16 v17, v18 offset:1024                          // 0000000003BC: D87C0400 00001211 
	s_and_saveexec_b32 s9, s28                                 // 0000000003C4: BE89201C 
	s_cbranch_execz 65474                                      // 0000000003C8: BFA5FFC2  <_amdgpu_cs_main+0x2D4>
	v_cndmask_b32_e64 v18, v29, v28, s7                        // 0000000003CC: D5010012 001E391D 
	v_cndmask_b32_e64 v27, v28, v29, s7                        // 0000000003D4: D501001B 001E3B1C 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000003DC: BF870092 
	v_mul_lo_u32 v18, v18, s5                                  // 0000000003E0: D72C0012 00000B12 
	v_add_lshl_u32 v18, v18, v27, 1                            // 0000000003E8: D6470012 02063712 
	s_waitcnt lgkmcnt(0)                                       // 0000000003F0: BF89FC07 
	buffer_load_u16 v27, v18, s[16:19], 0 offen                // 0000000003F4: E0480000 80441B12 
	s_branch 65461                                             // 0000000003FC: BFA0FFB5  <_amdgpu_cs_main+0x2D4>
	v_lshrrev_b32_e32 v16, 28, v21                             // 000000000400: 32202A9C 
	v_and_b32_e32 v17, 0x7ffffff0, v19                         // 000000000404: 362226FF 7FFFFFF0 
	v_cmp_gt_u32_e32 vcc_lo, 0x100, v19                        // 00000000040C: 7C9826FF 00000100 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000414: BF870093 
	v_add_nc_u32_e32 v16, v19, v16                             // 000000000418: 4A202113 
	v_and_b32_e32 v16, 0x7ffffff0, v16                         // 00000000041C: 362020FF 7FFFFFF0 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000424: BF870091 
	v_sub_nc_u32_e32 v16, v19, v16                             // 000000000428: 4C202113 
	v_lshlrev_b32_e32 v16, 1, v16                              // 00000000042C: 30202081 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000430: BF870001 
	v_lshl_add_u32 v17, v17, 1, v16                            // 000000000434: D6460011 04410311 
	ds_store_b16 v17, v8 offset:2048                           // 00000000043C: D87C0800 00000811 
	ds_store_b16 v17, v9 offset:2112                           // 000000000444: D87C0840 00000911 
	ds_store_b16 v17, v10 offset:2176                          // 00000000044C: D87C0880 00000A11 
	ds_store_b16 v17, v11 offset:2240                          // 000000000454: D87C08C0 00000B11 
	ds_store_b16 v17, v12 offset:2304                          // 00000000045C: D87C0900 00000C11 
	ds_store_b16 v17, v13 offset:2368                          // 000000000464: D87C0940 00000D11 
	ds_store_b16 v17, v14 offset:2432                          // 00000000046C: D87C0980 00000E11 
	ds_store_b16 v17, v15 offset:2496                          // 000000000474: D87C09C0 00000F11 
	v_add_nc_u32_e32 v16, 0x800, v17                           // 00000000047C: 4A2022FF 00000800 
	s_waitcnt lgkmcnt(0)                                       // 000000000484: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 000000000488: BE84206A 
	s_cbranch_execz 53                                         // 00000000048C: BFA50035  <_amdgpu_cs_main+0x564>
	v_lshrrev_b32_e32 v17, 4, v19                              // 000000000490: 32222684 
	v_or_b32_e32 v18, s10, v20                                 // 000000000494: 3824280A 
	v_lshl_add_u32 v21, v19, 1, 0x800                          // 000000000498: D6460015 03FD0313 00000800 
	s_lshl_b32 s5, s6, 3                                       // 0000000004A4: 84058306 
	s_mov_b32 s7, 0                                            // 0000000004A8: BE870080 
	v_add_nc_u32_e32 v17, s11, v17                             // 0000000004AC: 4A22220B 
	v_cmp_gt_u32_e64 s0, s2, v18                               // 0000000004B0: D44C0000 00022402 
	v_subrev_nc_u32_e32 v18, 32, v19                           // 0000000004B8: 4E2426A0 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000004BC: BF870093 
	v_mad_u64_u32 v[22:23], null, s6, v17, s[10:11]            // 0000000004C0: D6FE7C16 002A2206 
	v_add_lshl_u32 v22, v22, v20, 2                            // 0000000004C8: D6470016 020A2916 
	s_branch 23                                                // 0000000004D0: BFA00017  <_amdgpu_cs_main+0x530>
	s_nop 0                                                    // 0000000004D4: BF800000 
	s_nop 0                                                    // 0000000004D8: BF800000 
	s_nop 0                                                    // 0000000004DC: BF800000 
	s_nop 0                                                    // 0000000004E0: BF800000 
	s_nop 0                                                    // 0000000004E4: BF800000 
	s_nop 0                                                    // 0000000004E8: BF800000 
	s_nop 0                                                    // 0000000004EC: BF800000 
	s_nop 0                                                    // 0000000004F0: BF800000 
	s_nop 0                                                    // 0000000004F4: BF800000 
	s_nop 0                                                    // 0000000004F8: BF800000 
	s_nop 0                                                    // 0000000004FC: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000500: 8C7E037E 
	v_add_nc_u32_e32 v18, 32, v18                              // 000000000504: 4A2424A0 
	v_add_nc_u32_e32 v21, 64, v21                              // 000000000508: 4A2A2AC0 
	v_add_nc_u32_e32 v22, s5, v22                              // 00000000050C: 4A2C2C05 
	v_add_nc_u32_e32 v17, 2, v17                               // 000000000510: 4A222282 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000514: BF8704A4 
	v_cmp_lt_u32_e64 s3, 0xdf, v18                             // 000000000518: D4490003 000224FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000000524: 8C070703 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 000000000528: 917E077E 
	s_cbranch_execz 13                                         // 00000000052C: BFA5000D  <_amdgpu_cs_main+0x564>
	v_cmp_gt_u32_e64 s3, s1, v17                               // 000000000530: D44C0003 00022201 
	s_and_b32 s8, s3, s0                                       // 000000000538: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000053C: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000000540: BE832008 
	s_cbranch_execz 65518                                      // 000000000544: BFA5FFEE  <_amdgpu_cs_main+0x500>
	ds_load_u16_d16 v23, v21                                   // 000000000548: DA980000 17000015 
	s_waitcnt lgkmcnt(0)                                       // 000000000550: BF89FC07 
	v_cvt_f32_f16_e32 v23, v23                                 // 000000000554: 7E2E1717 
	buffer_store_b32 v23, v22, s[20:23], 0 offen               // 000000000558: E0680000 80451716 
	s_branch 65511                                             // 000000000560: BFA0FFE7  <_amdgpu_cs_main+0x500>
	s_or_b32 exec_lo, exec_lo, s4                              // 000000000564: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 000000000568: BC7C0000 
	ds_store_b16_d16_hi v16, v8                                // 00000000056C: DA840000 00000810 
	ds_store_b16_d16_hi v16, v9 offset:64                      // 000000000574: DA840040 00000910 
	ds_store_b16_d16_hi v16, v10 offset:128                    // 00000000057C: DA840080 00000A10 
	ds_store_b16_d16_hi v16, v11 offset:192                    // 000000000584: DA8400C0 00000B10 
	ds_store_b16_d16_hi v16, v12 offset:256                    // 00000000058C: DA840100 00000C10 
	ds_store_b16_d16_hi v16, v13 offset:320                    // 000000000594: DA840140 00000D10 
	ds_store_b16_d16_hi v16, v14 offset:384                    // 00000000059C: DA840180 00000E10 
	ds_store_b16_d16_hi v16, v15 offset:448                    // 0000000005A4: DA8401C0 00000F10 
	s_waitcnt lgkmcnt(0)                                       // 0000000005AC: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 0000000005B0: BE84206A 
	s_cbranch_execz 59                                         // 0000000005B4: BFA5003B  <_amdgpu_cs_main+0x6A4>
	v_lshrrev_b32_e32 v8, 4, v19                               // 0000000005B8: 32102684 
	v_or3_b32 v10, s10, v20, 16                                // 0000000005BC: D658000A 0242280A 
	s_lshl_b32 s5, s6, 3                                       // 0000000005C4: 84058306 
	s_mov_b32 s7, 0                                            // 0000000005C8: BE870080 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000005CC: BF870112 
	v_add_nc_u32_e32 v8, s11, v8                               // 0000000005D0: 4A10100B 
	v_cmp_gt_u32_e64 s0, s2, v10                               // 0000000005D4: D44C0000 00021402 
	v_lshl_add_u32 v10, v19, 1, 0x800                          // 0000000005DC: D646000A 03FD0313 00000800 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000005E8: BF870093 
	v_mul_lo_u32 v9, s6, v8                                    // 0000000005EC: D72C0009 00021006 
	v_add3_u32 v11, v9, s10, v20                               // 0000000005F4: D655000B 04501509 
	v_subrev_nc_u32_e32 v9, 32, v19                            // 0000000005FC: 4E1226A0 
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000600: BF870002 
	v_lshl_add_u32 v11, v11, 2, 64                             // 000000000604: D646000B 0301050B 
	s_branch 24                                                // 00000000060C: BFA00018  <_amdgpu_cs_main+0x670>
	s_nop 0                                                    // 000000000610: BF800000 
	s_nop 0                                                    // 000000000614: BF800000 
	s_nop 0                                                    // 000000000618: BF800000 
	s_nop 0                                                    // 00000000061C: BF800000 
	s_nop 0                                                    // 000000000620: BF800000 
	s_nop 0                                                    // 000000000624: BF800000 
	s_nop 0                                                    // 000000000628: BF800000 
	s_nop 0                                                    // 00000000062C: BF800000 
	s_nop 0                                                    // 000000000630: BF800000 
	s_nop 0                                                    // 000000000634: BF800000 
	s_nop 0                                                    // 000000000638: BF800000 
	s_nop 0                                                    // 00000000063C: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000640: 8C7E037E 
	v_add_nc_u32_e32 v9, 32, v9                                // 000000000644: 4A1212A0 
	v_add_nc_u32_e32 v10, 64, v10                              // 000000000648: 4A1414C0 
	v_add_nc_u32_e32 v11, s5, v11                              // 00000000064C: 4A161605 
	v_add_nc_u32_e32 v8, 2, v8                                 // 000000000650: 4A101082 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000654: BF8704A4 
	v_cmp_lt_u32_e64 s3, 0xdf, v9                              // 000000000658: D4490003 000212FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000000664: 8C070703 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 000000000668: 917E077E 
	s_cbranch_execz 13                                         // 00000000066C: BFA5000D  <_amdgpu_cs_main+0x6A4>
	v_cmp_gt_u32_e64 s3, s1, v8                                // 000000000670: D44C0003 00021001 
	s_and_b32 s8, s3, s0                                       // 000000000678: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000067C: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000000680: BE832008 
	s_cbranch_execz 65518                                      // 000000000684: BFA5FFEE  <_amdgpu_cs_main+0x640>
	ds_load_u16_d16 v12, v10                                   // 000000000688: DA980000 0C00000A 
	s_waitcnt lgkmcnt(0)                                       // 000000000690: BF89FC07 
	v_cvt_f32_f16_e32 v12, v12                                 // 000000000694: 7E18170C 
	buffer_store_b32 v12, v11, s[20:23], 0 offen               // 000000000698: E0680000 80450C0B 
	s_branch 65511                                             // 0000000006A0: BFA0FFE7  <_amdgpu_cs_main+0x640>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000006A4: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000006A8: BC7C0000 
	ds_store_b16 v16, v0                                       // 0000000006AC: D87C0000 00000010 
	ds_store_b16 v16, v1 offset:64                             // 0000000006B4: D87C0040 00000110 
	ds_store_b16 v16, v2 offset:128                            // 0000000006BC: D87C0080 00000210 
	ds_store_b16 v16, v3 offset:192                            // 0000000006C4: D87C00C0 00000310 
	ds_store_b16 v16, v4 offset:256                            // 0000000006CC: D87C0100 00000410 
	ds_store_b16 v16, v5 offset:320                            // 0000000006D4: D87C0140 00000510 
	ds_store_b16 v16, v6 offset:384                            // 0000000006DC: D87C0180 00000610 
	ds_store_b16 v16, v7 offset:448                            // 0000000006E4: D87C01C0 00000710 
	s_waitcnt lgkmcnt(0)                                       // 0000000006EC: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 0000000006F0: BE84206A 
	s_cbranch_execz 43                                         // 0000000006F4: BFA5002B  <_amdgpu_cs_main+0x7A4>
	v_lshrrev_b32_e32 v8, 4, v19                               // 0000000006F8: 32102684 
	v_or_b32_e32 v9, s10, v20                                  // 0000000006FC: 3812280A 
	v_lshl_add_u32 v10, v19, 1, 0x800                          // 000000000700: D646000A 03FD0313 00000800 
	s_lshl_b32 s5, s6, 3                                       // 00000000070C: 84058306 
	s_mov_b32 s7, 0                                            // 000000000710: BE870080 
	v_add3_u32 v8, v8, s11, 16                                 // 000000000714: D6550008 02401708 
	v_cmp_gt_u32_e64 s0, s2, v9                                // 00000000071C: D44C0000 00021202 
	v_subrev_nc_u32_e32 v9, 32, v19                            // 000000000724: 4E1226A0 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000728: BF870093 
	v_mad_u64_u32 v[11:12], null, s6, v8, s[10:11]             // 00000000072C: D6FE7C0B 002A1006 
	v_add_lshl_u32 v11, v11, v20, 2                            // 000000000734: D647000B 020A290B 
	s_branch 12                                                // 00000000073C: BFA0000C  <_amdgpu_cs_main+0x770>
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000740: 8C7E037E 
	v_add_nc_u32_e32 v9, 32, v9                                // 000000000744: 4A1212A0 
	v_add_nc_u32_e32 v10, 64, v10                              // 000000000748: 4A1414C0 
	v_add_nc_u32_e32 v8, 2, v8                                 // 00000000074C: 4A101082 
	v_add_nc_u32_e32 v11, s5, v11                              // 000000000750: 4A161605 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000754: BF8704A4 
	v_cmp_lt_u32_e64 s3, 0xdf, v9                              // 000000000758: D4490003 000212FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000000764: 8C070703 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 000000000768: 917E077E 
	s_cbranch_execz 13                                         // 00000000076C: BFA5000D  <_amdgpu_cs_main+0x7A4>
	v_cmp_gt_u32_e64 s3, s1, v8                                // 000000000770: D44C0003 00021001 
	s_and_b32 s8, s3, s0                                       // 000000000778: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000077C: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000000780: BE832008 
	s_cbranch_execz 65518                                      // 000000000784: BFA5FFEE  <_amdgpu_cs_main+0x740>
	ds_load_u16_d16 v12, v10                                   // 000000000788: DA980000 0C00000A 
	s_waitcnt lgkmcnt(0)                                       // 000000000790: BF89FC07 
	v_cvt_f32_f16_e32 v12, v12                                 // 000000000794: 7E18170C 
	buffer_store_b32 v12, v11, s[20:23], 0 offen               // 000000000798: E0680000 80450C0B 
	s_branch 65511                                             // 0000000007A0: BFA0FFE7  <_amdgpu_cs_main+0x740>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000007A4: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000007A8: BC7C0000 
	ds_store_b16_d16_hi v16, v0                                // 0000000007AC: DA840000 00000010 
	ds_store_b16_d16_hi v16, v1 offset:64                      // 0000000007B4: DA840040 00000110 
	ds_store_b16_d16_hi v16, v2 offset:128                     // 0000000007BC: DA840080 00000210 
	ds_store_b16_d16_hi v16, v3 offset:192                     // 0000000007C4: DA8400C0 00000310 
	ds_store_b16_d16_hi v16, v4 offset:256                     // 0000000007CC: DA840100 00000410 
	ds_store_b16_d16_hi v16, v5 offset:320                     // 0000000007D4: DA840140 00000510 
	ds_store_b16_d16_hi v16, v6 offset:384                     // 0000000007DC: DA840180 00000610 
	ds_store_b16_d16_hi v16, v7 offset:448                     // 0000000007E4: DA8401C0 00000710 
	s_waitcnt lgkmcnt(0)                                       // 0000000007EC: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000007F0: BE83206A 
	s_cbranch_execz 59                                         // 0000000007F4: BFA5003B  <_amdgpu_cs_main+0x8E4>
	v_lshrrev_b32_e32 v0, 4, v19                               // 0000000007F8: 32002684 
	v_or3_b32 v2, s10, v20, 16                                 // 0000000007FC: D6580002 0242280A 
	s_mov_b32 s4, 0                                            // 000000000804: BE840080 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000808: BF870112 
	v_add3_u32 v0, v0, s11, 16                                 // 00000000080C: D6550000 02401700 
	v_cmp_gt_u32_e32 vcc_lo, s2, v2                            // 000000000814: 7C980402 
	v_lshl_add_u32 v2, v19, 1, 0x800                           // 000000000818: D6460002 03FD0313 00000800 
	s_lshl_b32 s2, s6, 3                                       // 000000000824: 84028306 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000828: BF870093 
	v_mul_lo_u32 v1, s6, v0                                    // 00000000082C: D72C0001 00020006 
	v_add3_u32 v3, v1, s10, v20                                // 000000000834: D6550003 04501501 
	v_subrev_nc_u32_e32 v1, 32, v19                            // 00000000083C: 4E0226A0 
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000840: BF870002 
	v_lshl_add_u32 v3, v3, 2, 64                               // 000000000844: D6460003 03010503 
	s_branch 24                                                // 00000000084C: BFA00018  <_amdgpu_cs_main+0x8B0>
	s_nop 0                                                    // 000000000850: BF800000 
	s_nop 0                                                    // 000000000854: BF800000 
	s_nop 0                                                    // 000000000858: BF800000 
	s_nop 0                                                    // 00000000085C: BF800000 
	s_nop 0                                                    // 000000000860: BF800000 
	s_nop 0                                                    // 000000000864: BF800000 
	s_nop 0                                                    // 000000000868: BF800000 
	s_nop 0                                                    // 00000000086C: BF800000 
	s_nop 0                                                    // 000000000870: BF800000 
	s_nop 0                                                    // 000000000874: BF800000 
	s_nop 0                                                    // 000000000878: BF800000 
	s_nop 0                                                    // 00000000087C: BF800000 
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000880: 8C7E007E 
	v_add_nc_u32_e32 v1, 32, v1                                // 000000000884: 4A0202A0 
	v_add_nc_u32_e32 v2, 64, v2                                // 000000000888: 4A0404C0 
	v_add_nc_u32_e32 v0, 2, v0                                 // 00000000088C: 4A000082 
	v_add_nc_u32_e32 v3, s2, v3                                // 000000000890: 4A060602 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000894: BF8704A4 
	v_cmp_lt_u32_e64 s0, 0xdf, v1                              // 000000000898: D4490000 000202FF 000000DF 
	s_or_b32 s4, s0, s4                                        // 0000000008A4: 8C040400 
	s_and_not1_b32 exec_lo, exec_lo, s4                        // 0000000008A8: 917E047E 
	s_cbranch_execz 13                                         // 0000000008AC: BFA5000D  <_amdgpu_cs_main+0x8E4>
	v_cmp_gt_u32_e64 s0, s1, v0                                // 0000000008B0: D44C0000 00020001 
	s_and_b32 s5, s0, vcc_lo                                   // 0000000008B8: 8B056A00 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000008BC: BF870009 
	s_and_saveexec_b32 s0, s5                                  // 0000000008C0: BE802005 
	s_cbranch_execz 65518                                      // 0000000008C4: BFA5FFEE  <_amdgpu_cs_main+0x880>
	ds_load_u16_d16 v4, v2                                     // 0000000008C8: DA980000 04000002 
	s_waitcnt lgkmcnt(0)                                       // 0000000008D0: BF89FC07 
	v_cvt_f32_f16_e32 v4, v4                                   // 0000000008D4: 7E081704 
	buffer_store_b32 v4, v3, s[20:23], 0 offen                 // 0000000008D8: E0680000 80450403 
	s_branch 65511                                             // 0000000008E0: BFA0FFE7  <_amdgpu_cs_main+0x880>
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000008E4: 8C7E037E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000008E8: BC7C0000 
	s_endpgm                                                   // 0000000008EC: BFB00000 
	s_code_end                                                 // 0000000008F0: BF9F0000 
	s_code_end                                                 // 0000000008F4: BF9F0000 
	s_code_end                                                 // 0000000008F8: BF9F0000 
	s_code_end                                                 // 0000000008FC: BF9F0000 
	s_code_end                                                 // 000000000900: BF9F0000 
	s_code_end                                                 // 000000000904: BF9F0000 
	s_code_end                                                 // 000000000908: BF9F0000 
	s_code_end                                                 // 00000000090C: BF9F0000 
	s_code_end                                                 // 000000000910: BF9F0000 
	s_code_end                                                 // 000000000914: BF9F0000 
	s_code_end                                                 // 000000000918: BF9F0000 
	s_code_end                                                 // 00000000091C: BF9F0000 
	s_code_end                                                 // 000000000920: BF9F0000 
	s_code_end                                                 // 000000000924: BF9F0000 
	s_code_end                                                 // 000000000928: BF9F0000 
	s_code_end                                                 // 00000000092C: BF9F0000 
	s_code_end                                                 // 000000000930: BF9F0000 
	s_code_end                                                 // 000000000934: BF9F0000 
	s_code_end                                                 // 000000000938: BF9F0000 
	s_code_end                                                 // 00000000093C: BF9F0000 
	s_code_end                                                 // 000000000940: BF9F0000 
	s_code_end                                                 // 000000000944: BF9F0000 
	s_code_end                                                 // 000000000948: BF9F0000 
	s_code_end                                                 // 00000000094C: BF9F0000 
	s_code_end                                                 // 000000000950: BF9F0000 
	s_code_end                                                 // 000000000954: BF9F0000 
	s_code_end                                                 // 000000000958: BF9F0000 
	s_code_end                                                 // 00000000095C: BF9F0000 
	s_code_end                                                 // 000000000960: BF9F0000 
	s_code_end                                                 // 000000000964: BF9F0000 
	s_code_end                                                 // 000000000968: BF9F0000 
	s_code_end                                                 // 00000000096C: BF9F0000 
	s_code_end                                                 // 000000000970: BF9F0000 
	s_code_end                                                 // 000000000974: BF9F0000 
	s_code_end                                                 // 000000000978: BF9F0000 
	s_code_end                                                 // 00000000097C: BF9F0000 
	s_code_end                                                 // 000000000980: BF9F0000 
	s_code_end                                                 // 000000000984: BF9F0000 
	s_code_end                                                 // 000000000988: BF9F0000 
	s_code_end                                                 // 00000000098C: BF9F0000 
	s_code_end                                                 // 000000000990: BF9F0000 
	s_code_end                                                 // 000000000994: BF9F0000 
	s_code_end                                                 // 000000000998: BF9F0000 
	s_code_end                                                 // 00000000099C: BF9F0000 
	s_code_end                                                 // 0000000009A0: BF9F0000 
	s_code_end                                                 // 0000000009A4: BF9F0000 
	s_code_end                                                 // 0000000009A8: BF9F0000 
	s_code_end                                                 // 0000000009AC: BF9F0000 
	s_code_end                                                 // 0000000009B0: BF9F0000 
	s_code_end                                                 // 0000000009B4: BF9F0000 
	s_code_end                                                 // 0000000009B8: BF9F0000 
	s_code_end                                                 // 0000000009BC: BF9F0000 
	s_code_end                                                 // 0000000009C0: BF9F0000 
	s_code_end                                                 // 0000000009C4: BF9F0000 
	s_code_end                                                 // 0000000009C8: BF9F0000 
	s_code_end                                                 // 0000000009CC: BF9F0000 
	s_code_end                                                 // 0000000009D0: BF9F0000 
	s_code_end                                                 // 0000000009D4: BF9F0000 
	s_code_end                                                 // 0000000009D8: BF9F0000 
	s_code_end                                                 // 0000000009DC: BF9F0000 
	s_code_end                                                 // 0000000009E0: BF9F0000 
	s_code_end                                                 // 0000000009E4: BF9F0000 
	s_code_end                                                 // 0000000009E8: BF9F0000 
	s_code_end                                                 // 0000000009EC: BF9F0000 
	s_code_end                                                 // 0000000009F0: BF9F0000 
	s_code_end                                                 // 0000000009F4: BF9F0000 
	s_code_end                                                 // 0000000009F8: BF9F0000 
	s_code_end                                                 // 0000000009FC: BF9F0000 
	s_code_end                                                 // 000000000A00: BF9F0000 
	s_code_end                                                 // 000000000A04: BF9F0000 
	s_code_end                                                 // 000000000A08: BF9F0000 
	s_code_end                                                 // 000000000A0C: BF9F0000 
	s_code_end                                                 // 000000000A10: BF9F0000 
	s_code_end                                                 // 000000000A14: BF9F0000 
	s_code_end                                                 // 000000000A18: BF9F0000 
	s_code_end                                                 // 000000000A1C: BF9F0000 
	s_code_end                                                 // 000000000A20: BF9F0000 
	s_code_end                                                 // 000000000A24: BF9F0000 
	s_code_end                                                 // 000000000A28: BF9F0000 
	s_code_end                                                 // 000000000A2C: BF9F0000 
	s_code_end                                                 // 000000000A30: BF9F0000 
	s_code_end                                                 // 000000000A34: BF9F0000 
	s_code_end                                                 // 000000000A38: BF9F0000 
	s_code_end                                                 // 000000000A3C: BF9F0000 
	s_code_end                                                 // 000000000A40: BF9F0000 
	s_code_end                                                 // 000000000A44: BF9F0000 
	s_code_end                                                 // 000000000A48: BF9F0000 
	s_code_end                                                 // 000000000A4C: BF9F0000 
	s_code_end                                                 // 000000000A50: BF9F0000 
	s_code_end                                                 // 000000000A54: BF9F0000 
	s_code_end                                                 // 000000000A58: BF9F0000 
	s_code_end                                                 // 000000000A5C: BF9F0000 
	s_code_end                                                 // 000000000A60: BF9F0000 
	s_code_end                                                 // 000000000A64: BF9F0000 
	s_code_end                                                 // 000000000A68: BF9F0000 
	s_code_end                                                 // 000000000A6C: BF9F0000 
	s_code_end                                                 // 000000000A70: BF9F0000 
	s_code_end                                                 // 000000000A74: BF9F0000 
	s_code_end                                                 // 000000000A78: BF9F0000 
	s_code_end                                                 // 000000000A7C: BF9F0000 
