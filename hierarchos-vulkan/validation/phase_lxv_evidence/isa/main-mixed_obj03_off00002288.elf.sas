
:	file format elf64-amdgpu

Disassembly of section .text:
_amdgpu_cs_main:
	s_getpc_b64 s[12:13]                                       // 000000000000: BE8C4700 
	s_mov_b32 s0, s9                                           // 000000000004: BE800009 
	s_mov_b32 s9, s13                                          // 000000000008: BE89000D 
	v_mbcnt_lo_u32_b32 v35, -1, 0                              // 00000000000C: D71F0023 000100C1 
	s_load_b128 s[20:23], s[8:9], 0x20                         // 000000000014: F4080504 F8000020 
	v_mov_b32_e32 v31, 0                                       // 00000000001C: 7E3E0280 
	s_lshl_b32 s11, s10, 5                                     // 000000000020: 840B850A 
	s_lshl_b32 s10, s0, 5                                      // 000000000024: 840A8500 
	v_ashrrev_i32_e32 v37, 31, v35                             // 000000000028: 344A469F 
	v_and_b32_e32 v36, 15, v35                                 // 00000000002C: 3648468F 
	v_mov_b32_e32 v30, v31                                     // 000000000030: 7E3C031F 
	v_mov_b32_e32 v29, v31                                     // 000000000034: 7E3A031F 
	v_mov_b32_e32 v28, v31                                     // 000000000038: 7E38031F 
	v_mov_b32_e32 v27, v31                                     // 00000000003C: 7E36031F 
	v_mov_b32_e32 v26, v31                                     // 000000000040: 7E34031F 
	v_mov_b32_e32 v25, v31                                     // 000000000044: 7E32031F 
	v_mov_b32_e32 v24, v31                                     // 000000000048: 7E30031F 
	v_mov_b32_e32 v23, v31                                     // 00000000004C: 7E2E031F 
	v_mov_b32_e32 v22, v31                                     // 000000000050: 7E2C031F 
	v_mov_b32_e32 v21, v31                                     // 000000000054: 7E2A031F 
	v_mov_b32_e32 v20, v31                                     // 000000000058: 7E28031F 
	v_mov_b32_e32 v19, v31                                     // 00000000005C: 7E26031F 
	v_mov_b32_e32 v18, v31                                     // 000000000060: 7E24031F 
	v_mov_b32_e32 v17, v31                                     // 000000000064: 7E22031F 
	v_mov_b32_e32 v16, v31                                     // 000000000068: 7E20031F 
	v_mov_b32_e32 v15, v31                                     // 00000000006C: 7E1E031F 
	v_mov_b32_e32 v14, v31                                     // 000000000070: 7E1C031F 
	v_mov_b32_e32 v13, v31                                     // 000000000074: 7E1A031F 
	v_mov_b32_e32 v12, v31                                     // 000000000078: 7E18031F 
	v_mov_b32_e32 v11, v31                                     // 00000000007C: 7E16031F 
	v_mov_b32_e32 v10, v31                                     // 000000000080: 7E14031F 
	v_mov_b32_e32 v9, v31                                      // 000000000084: 7E12031F 
	v_mov_b32_e32 v8, v31                                      // 000000000088: 7E10031F 
	v_mov_b32_e32 v7, v31                                      // 00000000008C: 7E0E031F 
	v_mov_b32_e32 v6, v31                                      // 000000000090: 7E0C031F 
	v_mov_b32_e32 v5, v31                                      // 000000000094: 7E0A031F 
	v_mov_b32_e32 v4, v31                                      // 000000000098: 7E08031F 
	v_mov_b32_e32 v3, v31                                      // 00000000009C: 7E06031F 
	v_mov_b32_e32 v2, v31                                      // 0000000000A0: 7E04031F 
	v_mov_b32_e32 v1, v31                                      // 0000000000A4: 7E02031F 
	v_mov_b32_e32 v0, v31                                      // 0000000000A8: 7E00031F 
	s_cmp_eq_u32 s3, 0                                         // 0000000000AC: BF068003 
	s_mov_b32 s24, 0                                           // 0000000000B0: BE980080 
	s_cbranch_scc1 241                                         // 0000000000B4: BFA200F1  <_amdgpu_cs_main+0x47C>
	v_lshrrev_b32_e32 v0, 28, v37                              // 0000000000B8: 32004A9C 
	s_load_b256 s[12:19], s[8:9], null                         // 0000000000BC: F40C0304 F8000000 
	v_mov_b32_e32 v24, 0                                       // 0000000000C4: 7E300280 
	s_bitcmp1_b32 s7, 0                                        // 0000000000C8: BF0D8007 
	v_cmp_gt_u32_e64 s0, 0x200, v35                            // 0000000000CC: D44C0000 000246FF 00000200 
	v_add_nc_u32_e32 v0, v35, v0                               // 0000000000D8: 4A000123 
	v_or_b32_e32 v38, s10, v36                                 // 0000000000DC: 384C480A 
	v_mov_b32_e32 v25, v24                                     // 0000000000E0: 7E320318 
	v_mov_b32_e32 v26, v24                                     // 0000000000E4: 7E340318 
	v_mov_b32_e32 v28, v24                                     // 0000000000E8: 7E380318 
	v_dual_mov_b32 v27, v24 :: v_dual_and_b32 v0, -16, v0      // 0000000000EC: CA240118 1B0000D0 
	v_mov_b32_e32 v30, v24                                     // 0000000000F4: 7E3C0318 
	v_mov_b32_e32 v31, v24                                     // 0000000000F8: 7E3E0318 
	v_mov_b32_e32 v16, v24                                     // 0000000000FC: 7E200318 
	s_delay_alu instid0(VALU_DEP_4)                            // 000000000100: BF870004 
	v_sub_nc_u32_e32 v0, v35, v0                               // 000000000104: 4C000123 
	v_mov_b32_e32 v29, v24                                     // 000000000108: 7E3A0318 
	v_mov_b32_e32 v17, v24                                     // 00000000010C: 7E220318 
	v_mov_b32_e32 v18, v24                                     // 000000000110: 7E240318 
	v_mov_b32_e32 v19, v24                                     // 000000000114: 7E260318 
	v_lshl_add_u32 v39, v0, 5, 0x400                           // 000000000118: D6460027 03FD0B00 00000400 
	v_lshlrev_b32_e32 v40, 1, v0                               // 000000000124: 30500081 
	v_mov_b32_e32 v20, v24                                     // 000000000128: 7E280318 
	v_mov_b32_e32 v21, v24                                     // 00000000012C: 7E2A0318 
	v_mov_b32_e32 v22, v24                                     // 000000000130: 7E2C0318 
	v_mov_b32_e32 v23, v24                                     // 000000000134: 7E2E0318 
	v_mov_b32_e32 v8, v24                                      // 000000000138: 7E100318 
	v_mov_b32_e32 v9, v24                                      // 00000000013C: 7E120318 
	v_mov_b32_e32 v10, v24                                     // 000000000140: 7E140318 
	v_mov_b32_e32 v11, v24                                     // 000000000144: 7E160318 
	v_mov_b32_e32 v12, v24                                     // 000000000148: 7E180318 
	v_mov_b32_e32 v13, v24                                     // 00000000014C: 7E1A0318 
	v_mov_b32_e32 v14, v24                                     // 000000000150: 7E1C0318 
	v_mov_b32_e32 v15, v24                                     // 000000000154: 7E1E0318 
	v_mov_b32_e32 v0, v24                                      // 000000000158: 7E000318 
	v_mov_b32_e32 v1, v24                                      // 00000000015C: 7E020318 
	v_mov_b32_e32 v2, v24                                      // 000000000160: 7E040318 
	v_mov_b32_e32 v3, v24                                      // 000000000164: 7E060318 
	v_mov_b32_e32 v4, v24                                      // 000000000168: 7E080318 
	v_mov_b32_e32 v5, v24                                      // 00000000016C: 7E0A0318 
	v_mov_b32_e32 v6, v24                                      // 000000000170: 7E0C0318 
	v_mov_b32_e32 v7, v24                                      // 000000000174: 7E0E0318 
	s_cselect_b32 s25, -1, 0                                   // 000000000178: 981980C1 
	s_bitcmp0_b32 s7, 1                                        // 00000000017C: BF0C8107 
	s_cselect_b32 s7, -1, 0                                    // 000000000180: 980780C1 
	s_branch 103                                               // 000000000184: BFA00067  <_amdgpu_cs_main+0x324>
	s_or_b32 exec_lo, exec_lo, s26                             // 000000000188: 8C7E1A7E 
	s_waitcnt lgkmcnt(0)                                       // 00000000018C: BF89FC07 
	ds_load_u16_d16 v41, v40                                   // 000000000190: DA980000 29000028 
	ds_load_u16_d16 v42, v40 offset:64                         // 000000000198: DA980040 2A000028 
	ds_load_u16_d16 v43, v40 offset:128                        // 0000000001A0: DA980080 2B000028 
	ds_load_u16_d16 v44, v40 offset:192                        // 0000000001A8: DA9800C0 2C000028 
	ds_load_u16_d16 v45, v40 offset:256                        // 0000000001B0: DA980100 2D000028 
	ds_load_u16_d16 v46, v40 offset:320                        // 0000000001B8: DA980140 2E000028 
	ds_load_u16_d16 v47, v40 offset:384                        // 0000000001C0: DA980180 2F000028 
	ds_load_u16_d16 v48, v40 offset:448                        // 0000000001C8: DA9801C0 30000028 
	ds_load_u16_d16 v49, v40 offset:512                        // 0000000001D0: DA980200 31000028 
	ds_load_u16_d16 v50, v40 offset:576                        // 0000000001D8: DA980240 32000028 
	ds_load_u16_d16 v51, v40 offset:640                        // 0000000001E0: DA980280 33000028 
	ds_load_u16_d16 v52, v40 offset:704                        // 0000000001E8: DA9802C0 34000028 
	ds_load_u16_d16 v53, v40 offset:768                        // 0000000001F0: DA980300 35000028 
	ds_load_u16_d16 v54, v40 offset:832                        // 0000000001F8: DA980340 36000028 
	ds_load_u16_d16 v55, v40 offset:896                        // 000000000200: DA980380 37000028 
	ds_load_u16_d16 v56, v40 offset:960                        // 000000000208: DA9803C0 38000028 
	ds_load_b128 v[57:60], v39                                 // 000000000210: DBFC0000 39000027 
	ds_load_b128 v[61:64], v39 offset:16                       // 000000000218: DBFC0010 3D000027 
	ds_load_b128 v[65:68], v39 offset:512                      // 000000000220: DBFC0200 41000027 
	ds_load_b128 v[69:72], v39 offset:528                      // 000000000228: DBFC0210 45000027 
	s_add_i32 s24, s24, 16                                     // 000000000230: 81189018 
	s_waitcnt lgkmcnt(19)                                      // 000000000234: BF89FD37 
	ds_load_u16_d16_hi v41, v40 offset:32                      // 000000000238: DA9C0020 29000028 
	s_waitcnt lgkmcnt(19)                                      // 000000000240: BF89FD37 
	ds_load_u16_d16_hi v42, v40 offset:96                      // 000000000244: DA9C0060 2A000028 
	s_waitcnt lgkmcnt(19)                                      // 00000000024C: BF89FD37 
	ds_load_u16_d16_hi v43, v40 offset:160                     // 000000000250: DA9C00A0 2B000028 
	s_waitcnt lgkmcnt(19)                                      // 000000000258: BF89FD37 
	ds_load_u16_d16_hi v44, v40 offset:224                     // 00000000025C: DA9C00E0 2C000028 
	s_waitcnt lgkmcnt(19)                                      // 000000000264: BF89FD37 
	ds_load_u16_d16_hi v45, v40 offset:288                     // 000000000268: DA9C0120 2D000028 
	s_waitcnt lgkmcnt(19)                                      // 000000000270: BF89FD37 
	ds_load_u16_d16_hi v46, v40 offset:352                     // 000000000274: DA9C0160 2E000028 
	s_waitcnt lgkmcnt(19)                                      // 00000000027C: BF89FD37 
	ds_load_u16_d16_hi v47, v40 offset:416                     // 000000000280: DA9C01A0 2F000028 
	s_waitcnt lgkmcnt(19)                                      // 000000000288: BF89FD37 
	ds_load_u16_d16_hi v48, v40 offset:480                     // 00000000028C: DA9C01E0 30000028 
	s_waitcnt lgkmcnt(19)                                      // 000000000294: BF89FD37 
	ds_load_u16_d16_hi v49, v40 offset:544                     // 000000000298: DA9C0220 31000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002A0: BF89FD37 
	ds_load_u16_d16_hi v50, v40 offset:608                     // 0000000002A4: DA9C0260 32000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002AC: BF89FD37 
	ds_load_u16_d16_hi v51, v40 offset:672                     // 0000000002B0: DA9C02A0 33000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002B8: BF89FD37 
	ds_load_u16_d16_hi v52, v40 offset:736                     // 0000000002BC: DA9C02E0 34000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002C4: BF89FD37 
	ds_load_u16_d16_hi v53, v40 offset:800                     // 0000000002C8: DA9C0320 35000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002D0: BF89FD37 
	ds_load_u16_d16_hi v54, v40 offset:864                     // 0000000002D4: DA9C0360 36000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002DC: BF89FD37 
	ds_load_u16_d16_hi v55, v40 offset:928                     // 0000000002E0: DA9C03A0 37000028 
	s_waitcnt lgkmcnt(19)                                      // 0000000002E8: BF89FD37 
	ds_load_u16_d16_hi v56, v40 offset:992                     // 0000000002EC: DA9C03E0 38000028 
	s_cmp_lt_u32 s24, s3                                       // 0000000002F4: BF0A0318 
	s_waitcnt lgkmcnt(8)                                       // 0000000002F8: BF89FC87 
	v_wmma_f32_16x16x16_f16 v[24:31], v[57:64], v[41:48], v[24:31]// 0000000002FC: CC404018 1C625339 
	v_wmma_f32_16x16x16_f16 v[8:15], v[65:72], v[41:48], v[8:15]// 000000000304: CC404008 1C225341 
	s_waitcnt lgkmcnt(0)                                       // 00000000030C: BF89FC07 
	v_wmma_f32_16x16x16_f16 v[16:23], v[57:64], v[49:56], v[16:23]// 000000000310: CC404010 1C426339 
	v_wmma_f32_16x16x16_f16 v[0:7], v[65:72], v[49:56], v[0:7] // 000000000318: CC404000 1C026341 
	s_cbranch_scc0 86                                          // 000000000320: BFA10056  <_amdgpu_cs_main+0x47C>
	s_and_saveexec_b32 s26, s0                                 // 000000000324: BE9A2000 
	s_cbranch_execz 65431                                      // 000000000328: BFA5FF97  <_amdgpu_cs_main+0x188>
	v_or_b32_e32 v32, s24, v36                                 // 00000000032C: 38404818 
	v_mov_b32_e32 v42, v35                                     // 000000000330: 7E540323 
	s_mov_b32 s27, 0                                           // 000000000334: BE9B0080 
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000338: BF870002 
	v_mul_lo_u32 v41, v32, s4                                  // 00000000033C: D72C0029 00000920 
	v_cmp_gt_u32_e64 s8, s3, v32                               // 000000000344: D44C0008 00024003 
	s_branch 12                                                // 00000000034C: BFA0000C  <_amdgpu_cs_main+0x380>
	s_or_b32 exec_lo, exec_lo, s9                              // 000000000350: 8C7E097E 
	v_add_nc_u32_e32 v34, 32, v42                              // 000000000354: 4A4454A0 
	v_cmp_lt_u32_e32 vcc_lo, 0x1df, v42                        // 000000000358: 7C9254FF 000001DF 
	s_waitcnt vmcnt(0)                                         // 000000000360: BF8903F7 
	ds_store_b16 v33, v43                                      // 000000000364: D87C0000 00002B21 
	v_mov_b32_e32 v42, v34                                     // 00000000036C: 7E540322 
	s_or_b32 s27, vcc_lo, s27                                  // 000000000370: 8C1B1B6A 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000374: BF870009 
	s_and_not1_b32 exec_lo, exec_lo, s27                       // 000000000378: 917E1B7E 
	s_cbranch_execz 65410                                      // 00000000037C: BFA5FF82  <_amdgpu_cs_main+0x188>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000000380: BF870241 
	v_lshrrev_b32_e32 v43, 8, v42                              // 000000000384: 32565488 
	v_bfe_u32 v44, v42, 4, 4                                   // 000000000388: D610002C 0211092A 
	v_and_b32_e32 v46, 0xff, v42                               // 000000000390: 365C54FF 000000FF 
 <unknown>
	v_lshlrev_b32_e32 v45, 4, v43                              // 00000000039C: 305A5684 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000003A0: BF870091 
	v_or3_b32 v47, v45, s11, v44                               // 0000000003A4: D658002F 04B0172D 
	v_cmp_gt_u32_e32 vcc_lo, s1, v47                           // 0000000003AC: 7C985E01 
	s_and_b32 s28, s8, vcc_lo                                  // 0000000003B0: 8B1C6A08 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000003B4: BF870009 
	s_and_saveexec_b32 s9, s28                                 // 0000000003B8: BE89201C 
	s_cbranch_execz 16                                         // 0000000003BC: BFA50010  <_amdgpu_cs_main+0x400>
	s_and_not1_b32 vcc_lo, exec_lo, s25                        // 0000000003C0: 916A197E 
	s_mov_b32 s28, -1                                          // 0000000003C4: BE9C00C1 
	s_cbranch_vccnz 2                                          // 0000000003C8: BFA40002  <_amdgpu_cs_main+0x3D4>
	v_add_nc_u32_e32 v33, v47, v41                             // 0000000003CC: 4A42532F 
	s_mov_b32 s28, 0                                           // 0000000003D0: BE9C0080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000003D4: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 0000000003D8: 916A1C7E 
	s_cbranch_vccnz 3                                          // 0000000003DC: BFA40003  <_amdgpu_cs_main+0x3EC>
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000003E0: BF870001 
	v_mad_u64_u32 v[33:34], null, v47, s4, v[32:33]            // 0000000003E4: D6FE7C21 0480092F 
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000003EC: BF870001 
	v_lshlrev_b32_e32 v33, 1, v33                              // 0000000003F0: 30424281 
	s_waitcnt lgkmcnt(0)                                       // 0000000003F4: BF89FC07 
	buffer_load_u16 v34, v33, s[12:15], 0 offen                // 0000000003F8: E0480000 80432221 
	s_or_b32 exec_lo, exec_lo, s9                              // 000000000400: 8C7E097E 
	v_or_b32_e32 v45, v38, v45                                 // 000000000404: 385A5B26 
	v_or_b32_e32 v44, s24, v44                                 // 000000000408: 38585818 
	v_lshlrev_b32_e32 v33, 1, v46                              // 00000000040C: 30425C81 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000000410: BF870193 
	v_cmp_gt_u32_e32 vcc_lo, s2, v45                           // 000000000414: 7C985A02 
	v_cmp_gt_u32_e64 s9, s3, v44                               // 000000000418: D44C0009 00025803 
	s_delay_alu instid0(VALU_DEP_3)                            // 000000000420: BF870003 
	v_lshl_or_b32 v33, v43, 9, v33                             // 000000000424: D6560021 0485132B 
 <unknown>
	s_and_b32 s28, s9, vcc_lo                                  // 000000000430: 8B1C6A09 
	s_waitcnt vmcnt(0)                                         // 000000000434: BF8903F7 
	ds_store_b16 v33, v34 offset:1024                          // 000000000438: D87C0400 00002221 
	s_and_saveexec_b32 s9, s28                                 // 000000000440: BE89201C 
	s_cbranch_execz 65474                                      // 000000000444: BFA5FFC2  <_amdgpu_cs_main+0x350>
	v_cndmask_b32_e64 v34, v45, v44, s7                        // 000000000448: D5010022 001E592D 
	v_cndmask_b32_e64 v43, v44, v45, s7                        // 000000000450: D501002B 001E5B2C 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000458: BF870092 
	v_mul_lo_u32 v34, v34, s5                                  // 00000000045C: D72C0022 00000B22 
	v_add_lshl_u32 v34, v34, v43, 1                            // 000000000464: D6470022 02065722 
	s_waitcnt lgkmcnt(0)                                       // 00000000046C: BF89FC07 
	buffer_load_u16 v43, v34, s[16:19], 0 offen                // 000000000470: E0480000 80442B22 
	s_branch 65461                                             // 000000000478: BFA0FFB5  <_amdgpu_cs_main+0x350>
	v_lshrrev_b32_e32 v32, 28, v37                             // 00000000047C: 32404A9C 
	v_and_b32_e32 v33, 0x3ffffff0, v35                         // 000000000480: 364246FF 3FFFFFF0 
	v_cmp_gt_u32_e32 vcc_lo, 0x100, v35                        // 000000000488: 7C9846FF 00000100 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000490: BF870093 
	v_add_nc_u32_e32 v32, v35, v32                             // 000000000494: 4A404123 
	v_and_b32_e32 v32, 0x3ffffff0, v32                         // 000000000498: 364040FF 3FFFFFF0 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000004A0: BF870091 
	v_sub_nc_u32_e32 v32, v35, v32                             // 0000000004A4: 4C404123 
	v_lshlrev_b32_e32 v32, 2, v32                              // 0000000004A8: 30404082 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000004AC: BF870091 
	v_lshl_add_u32 v32, v33, 2, v32                            // 0000000004B0: D6460020 04810521 
	v_add_nc_u32_e32 v33, 0x800, v32                           // 0000000004B8: 4A4240FF 00000800 
	v_add_nc_u32_e32 v32, 0x800, v32                           // 0000000004C0: 4A4040FF 00000800 
	ds_store_2addr_b32 v33, v24, v25 offset1:32                // 0000000004C8: D8382000 00191821 
	ds_store_2addr_b32 v33, v26, v27 offset0:64 offset1:96     // 0000000004D0: D8386040 001B1A21 
	ds_store_2addr_b32 v33, v28, v29 offset0:128 offset1:160   // 0000000004D8: D838A080 001D1C21 
	ds_store_2addr_b32 v33, v30, v31 offset0:192 offset1:224   // 0000000004E0: D838E0C0 001F1E21 
	s_waitcnt lgkmcnt(0)                                       // 0000000004E8: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 0000000004EC: BE84206A 
	s_cbranch_execz 44                                         // 0000000004F0: BFA5002C  <_amdgpu_cs_main+0x5A4>
	v_lshrrev_b32_e32 v24, 4, v35                              // 0000000004F4: 32304684 
	v_or_b32_e32 v25, s10, v36                                 // 0000000004F8: 3832480A 
	v_lshl_add_u32 v26, v35, 2, 0x800                          // 0000000004FC: D646001A 03FD0523 00000800 
	s_lshl_b32 s5, s6, 3                                       // 000000000508: 84058306 
	s_mov_b32 s7, 0                                            // 00000000050C: BE870080 
	v_add_nc_u32_e32 v24, s11, v24                             // 000000000510: 4A30300B 
	v_cmp_gt_u32_e64 s0, s2, v25                               // 000000000514: D44C0000 00023202 
	v_subrev_nc_u32_e32 v25, 32, v35                           // 00000000051C: 4E3246A0 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000520: BF870093 
	v_mad_u64_u32 v[27:28], null, s6, v24, s[10:11]            // 000000000524: D6FE7C1B 002A3006 
	v_add_lshl_u32 v27, v27, v36, 2                            // 00000000052C: D647001B 020A491B 
	s_branch 15                                                // 000000000534: BFA0000F  <_amdgpu_cs_main+0x574>
	s_nop 0                                                    // 000000000538: BF800000 
	s_nop 0                                                    // 00000000053C: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000540: 8C7E037E 
	v_add_nc_u32_e32 v25, 32, v25                              // 000000000544: 4A3232A0 
	v_add_nc_u32_e32 v26, 0x80, v26                            // 000000000548: 4A3434FF 00000080 
	v_add_nc_u32_e32 v27, s5, v27                              // 000000000550: 4A363605 
	v_add_nc_u32_e32 v24, 2, v24                               // 000000000554: 4A303082 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000558: BF8704A4 
	v_cmp_lt_u32_e64 s3, 0xdf, v25                             // 00000000055C: D4490003 000232FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000000568: 8C070703 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 00000000056C: 917E077E 
	s_cbranch_execz 12                                         // 000000000570: BFA5000C  <_amdgpu_cs_main+0x5A4>
	v_cmp_gt_u32_e64 s3, s1, v24                               // 000000000574: D44C0003 00023001 
	s_and_b32 s8, s3, s0                                       // 00000000057C: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000580: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000000584: BE832008 
	s_cbranch_execz 65517                                      // 000000000588: BFA5FFED  <_amdgpu_cs_main+0x540>
	ds_load_b32 v28, v26                                       // 00000000058C: D8D80000 1C00001A 
	s_waitcnt lgkmcnt(0)                                       // 000000000594: BF89FC07 
	buffer_store_b32 v28, v27, s[20:23], 0 offen               // 000000000598: E0680000 80451C1B 
	s_branch 65511                                             // 0000000005A0: BFA0FFE7  <_amdgpu_cs_main+0x540>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000005A4: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000005A8: BC7C0000 
	ds_store_2addr_b32 v32, v16, v17 offset1:32                // 0000000005AC: D8382000 00111020 
	ds_store_2addr_b32 v32, v18, v19 offset0:64 offset1:96     // 0000000005B4: D8386040 00131220 
	ds_store_2addr_b32 v32, v20, v21 offset0:128 offset1:160   // 0000000005BC: D838A080 00151420 
	ds_store_2addr_b32 v32, v22, v23 offset0:192 offset1:224   // 0000000005C4: D838E0C0 00171620 
	s_waitcnt lgkmcnt(0)                                       // 0000000005CC: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 0000000005D0: BE84206A 
	s_cbranch_execz 51                                         // 0000000005D4: BFA50033  <_amdgpu_cs_main+0x6A4>
	v_lshrrev_b32_e32 v16, 4, v35                              // 0000000005D8: 32204684 
	v_or3_b32 v18, s10, v36, 16                                // 0000000005DC: D6580012 0242480A 
	s_lshl_b32 s5, s6, 3                                       // 0000000005E4: 84058306 
	s_mov_b32 s7, 0                                            // 0000000005E8: BE870080 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000005EC: BF870112 
	v_add_nc_u32_e32 v16, s11, v16                             // 0000000005F0: 4A20200B 
	v_cmp_gt_u32_e64 s0, s2, v18                               // 0000000005F4: D44C0000 00022402 
	v_lshl_add_u32 v18, v35, 2, 0x800                          // 0000000005FC: D6460012 03FD0523 00000800 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000608: BF870093 
	v_mul_lo_u32 v17, s6, v16                                  // 00000000060C: D72C0011 00022006 
	v_add3_u32 v19, v17, s10, v36                              // 000000000614: D6550013 04901511 
	v_subrev_nc_u32_e32 v17, 32, v35                           // 00000000061C: 4E2246A0 
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000620: BF870002 
	v_lshl_add_u32 v19, v19, 2, 64                             // 000000000624: D6460013 03010513 
	s_branch 17                                                // 00000000062C: BFA00011  <_amdgpu_cs_main+0x674>
	s_nop 0                                                    // 000000000630: BF800000 
	s_nop 0                                                    // 000000000634: BF800000 
	s_nop 0                                                    // 000000000638: BF800000 
	s_nop 0                                                    // 00000000063C: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000640: 8C7E037E 
	v_add_nc_u32_e32 v17, 32, v17                              // 000000000644: 4A2222A0 
	v_add_nc_u32_e32 v18, 0x80, v18                            // 000000000648: 4A2424FF 00000080 
	v_add_nc_u32_e32 v19, s5, v19                              // 000000000650: 4A262605 
	v_add_nc_u32_e32 v16, 2, v16                               // 000000000654: 4A202082 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000658: BF8704A4 
	v_cmp_lt_u32_e64 s3, 0xdf, v17                             // 00000000065C: D4490003 000222FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000000668: 8C070703 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 00000000066C: 917E077E 
	s_cbranch_execz 12                                         // 000000000670: BFA5000C  <_amdgpu_cs_main+0x6A4>
	v_cmp_gt_u32_e64 s3, s1, v16                               // 000000000674: D44C0003 00022001 
	s_and_b32 s8, s3, s0                                       // 00000000067C: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000680: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000000684: BE832008 
	s_cbranch_execz 65517                                      // 000000000688: BFA5FFED  <_amdgpu_cs_main+0x640>
	ds_load_b32 v20, v18                                       // 00000000068C: D8D80000 14000012 
	s_waitcnt lgkmcnt(0)                                       // 000000000694: BF89FC07 
	buffer_store_b32 v20, v19, s[20:23], 0 offen               // 000000000698: E0680000 80451413 
	s_branch 65511                                             // 0000000006A0: BFA0FFE7  <_amdgpu_cs_main+0x640>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000006A4: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000006A8: BC7C0000 
	ds_store_2addr_b32 v32, v8, v9 offset1:32                  // 0000000006AC: D8382000 00090820 
	ds_store_2addr_b32 v32, v10, v11 offset0:64 offset1:96     // 0000000006B4: D8386040 000B0A20 
	ds_store_2addr_b32 v32, v12, v13 offset0:128 offset1:160   // 0000000006BC: D838A080 000D0C20 
	ds_store_2addr_b32 v32, v14, v15 offset0:192 offset1:224   // 0000000006C4: D838E0C0 000F0E20 
	s_waitcnt lgkmcnt(0)                                       // 0000000006CC: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 0000000006D0: BE84206A 
	s_cbranch_execz 51                                         // 0000000006D4: BFA50033  <_amdgpu_cs_main+0x7A4>
	v_lshrrev_b32_e32 v8, 4, v35                               // 0000000006D8: 32104684 
	v_or_b32_e32 v9, s10, v36                                  // 0000000006DC: 3812480A 
	v_lshl_add_u32 v10, v35, 2, 0x800                          // 0000000006E0: D646000A 03FD0523 00000800 
	s_lshl_b32 s5, s6, 3                                       // 0000000006EC: 84058306 
	s_mov_b32 s7, 0                                            // 0000000006F0: BE870080 
	v_add3_u32 v8, v8, s11, 16                                 // 0000000006F4: D6550008 02401708 
	v_cmp_gt_u32_e64 s0, s2, v9                                // 0000000006FC: D44C0000 00021202 
	v_subrev_nc_u32_e32 v9, 32, v35                            // 000000000704: 4E1246A0 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000708: BF870093 
	v_mad_u64_u32 v[11:12], null, s6, v8, s[10:11]             // 00000000070C: D6FE7C0B 002A1006 
	v_add_lshl_u32 v11, v11, v36, 2                            // 000000000714: D647000B 020A490B 
	s_branch 21                                                // 00000000071C: BFA00015  <_amdgpu_cs_main+0x774>
	s_nop 0                                                    // 000000000720: BF800000 
	s_nop 0                                                    // 000000000724: BF800000 
	s_nop 0                                                    // 000000000728: BF800000 
	s_nop 0                                                    // 00000000072C: BF800000 
	s_nop 0                                                    // 000000000730: BF800000 
	s_nop 0                                                    // 000000000734: BF800000 
	s_nop 0                                                    // 000000000738: BF800000 
	s_nop 0                                                    // 00000000073C: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000000740: 8C7E037E 
	v_add_nc_u32_e32 v9, 32, v9                                // 000000000744: 4A1212A0 
	v_add_nc_u32_e32 v10, 0x80, v10                            // 000000000748: 4A1414FF 00000080 
	v_add_nc_u32_e32 v8, 2, v8                                 // 000000000750: 4A101082 
	v_add_nc_u32_e32 v11, s5, v11                              // 000000000754: 4A161605 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000758: BF8704A4 
	v_cmp_lt_u32_e64 s3, 0xdf, v9                              // 00000000075C: D4490003 000212FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000000768: 8C070703 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 00000000076C: 917E077E 
	s_cbranch_execz 12                                         // 000000000770: BFA5000C  <_amdgpu_cs_main+0x7A4>
	v_cmp_gt_u32_e64 s3, s1, v8                                // 000000000774: D44C0003 00021001 
	s_and_b32 s8, s3, s0                                       // 00000000077C: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000780: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000000784: BE832008 
	s_cbranch_execz 65517                                      // 000000000788: BFA5FFED  <_amdgpu_cs_main+0x740>
	ds_load_b32 v12, v10                                       // 00000000078C: D8D80000 0C00000A 
	s_waitcnt lgkmcnt(0)                                       // 000000000794: BF89FC07 
	buffer_store_b32 v12, v11, s[20:23], 0 offen               // 000000000798: E0680000 80450C0B 
	s_branch 65511                                             // 0000000007A0: BFA0FFE7  <_amdgpu_cs_main+0x740>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000007A4: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000007A8: BC7C0000 
	ds_store_2addr_b32 v32, v0, v1 offset1:32                  // 0000000007AC: D8382000 00010020 
	ds_store_2addr_b32 v32, v2, v3 offset0:64 offset1:96       // 0000000007B4: D8386040 00030220 
	ds_store_2addr_b32 v32, v4, v5 offset0:128 offset1:160     // 0000000007BC: D838A080 00050420 
	ds_store_2addr_b32 v32, v6, v7 offset0:192 offset1:224     // 0000000007C4: D838E0C0 00070620 
	s_waitcnt lgkmcnt(0)                                       // 0000000007CC: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000007D0: BE83206A 
	s_cbranch_execz 51                                         // 0000000007D4: BFA50033  <_amdgpu_cs_main+0x8A4>
	v_lshrrev_b32_e32 v0, 4, v35                               // 0000000007D8: 32004684 
	v_or3_b32 v2, s10, v36, 16                                 // 0000000007DC: D6580002 0242480A 
	s_mov_b32 s4, 0                                            // 0000000007E4: BE840080 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000007E8: BF870112 
	v_add3_u32 v0, v0, s11, 16                                 // 0000000007EC: D6550000 02401700 
	v_cmp_gt_u32_e32 vcc_lo, s2, v2                            // 0000000007F4: 7C980402 
	v_lshl_add_u32 v2, v35, 2, 0x800                           // 0000000007F8: D6460002 03FD0523 00000800 
	s_lshl_b32 s2, s6, 3                                       // 000000000804: 84028306 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000808: BF870093 
	v_mul_lo_u32 v1, s6, v0                                    // 00000000080C: D72C0001 00020006 
	v_add3_u32 v3, v1, s10, v36                                // 000000000814: D6550003 04901501 
	v_subrev_nc_u32_e32 v1, 32, v35                            // 00000000081C: 4E0246A0 
	s_delay_alu instid0(VALU_DEP_2)                            // 000000000820: BF870002 
	v_lshl_add_u32 v3, v3, 2, 64                               // 000000000824: D6460003 03010503 
	s_branch 17                                                // 00000000082C: BFA00011  <_amdgpu_cs_main+0x874>
	s_nop 0                                                    // 000000000830: BF800000 
	s_nop 0                                                    // 000000000834: BF800000 
	s_nop 0                                                    // 000000000838: BF800000 
	s_nop 0                                                    // 00000000083C: BF800000 
	s_or_b32 exec_lo, exec_lo, s0                              // 000000000840: 8C7E007E 
	v_add_nc_u32_e32 v1, 32, v1                                // 000000000844: 4A0202A0 
	v_add_nc_u32_e32 v2, 0x80, v2                              // 000000000848: 4A0404FF 00000080 
	v_add_nc_u32_e32 v0, 2, v0                                 // 000000000850: 4A000082 
	v_add_nc_u32_e32 v3, s2, v3                                // 000000000854: 4A060602 
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000858: BF8704A4 
	v_cmp_lt_u32_e64 s0, 0xdf, v1                              // 00000000085C: D4490000 000202FF 000000DF 
	s_or_b32 s4, s0, s4                                        // 000000000868: 8C040400 
	s_and_not1_b32 exec_lo, exec_lo, s4                        // 00000000086C: 917E047E 
	s_cbranch_execz 12                                         // 000000000870: BFA5000C  <_amdgpu_cs_main+0x8A4>
	v_cmp_gt_u32_e64 s0, s1, v0                                // 000000000874: D44C0000 00020001 
	s_and_b32 s5, s0, vcc_lo                                   // 00000000087C: 8B056A00 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000880: BF870009 
	s_and_saveexec_b32 s0, s5                                  // 000000000884: BE802005 
	s_cbranch_execz 65517                                      // 000000000888: BFA5FFED  <_amdgpu_cs_main+0x840>
	ds_load_b32 v4, v2                                         // 00000000088C: D8D80000 04000002 
	s_waitcnt lgkmcnt(0)                                       // 000000000894: BF89FC07 
	buffer_store_b32 v4, v3, s[20:23], 0 offen                 // 000000000898: E0680000 80450403 
	s_branch 65511                                             // 0000000008A0: BFA0FFE7  <_amdgpu_cs_main+0x840>
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000008A4: 8C7E037E 
	s_waitcnt_vscnt null, 0x0                                  // 0000000008A8: BC7C0000 
	s_nop 0                                                    // 0000000008AC: BF800000 
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000008B0: BFB60003 
	s_endpgm                                                   // 0000000008B4: BFB00000 
	s_code_end                                                 // 0000000008B8: BF9F0000 
	s_code_end                                                 // 0000000008BC: BF9F0000 
	s_code_end                                                 // 0000000008C0: BF9F0000 
	s_code_end                                                 // 0000000008C4: BF9F0000 
	s_code_end                                                 // 0000000008C8: BF9F0000 
	s_code_end                                                 // 0000000008CC: BF9F0000 
	s_code_end                                                 // 0000000008D0: BF9F0000 
	s_code_end                                                 // 0000000008D4: BF9F0000 
	s_code_end                                                 // 0000000008D8: BF9F0000 
	s_code_end                                                 // 0000000008DC: BF9F0000 
	s_code_end                                                 // 0000000008E0: BF9F0000 
	s_code_end                                                 // 0000000008E4: BF9F0000 
	s_code_end                                                 // 0000000008E8: BF9F0000 
	s_code_end                                                 // 0000000008EC: BF9F0000 
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
