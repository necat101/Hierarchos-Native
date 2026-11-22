
:	file format elf64-amdgpu

Disassembly of section .text:
_amdgpu_cs_main:
	s_getpc_b64 s[12:13]                                       // 000000000000: BE8C4700 
	s_mov_b32 s0, s9                                           // 000000000004: BE800009 
	s_mov_b32 s9, s13                                          // 000000000008: BE89000D 
	v_mbcnt_lo_u32_b32 v129, -1, 0                             // 00000000000C: D71F0081 000100C1 
	s_load_b128 s[20:23], s[8:9], 0x20                         // 000000000014: F4080504 F8000020 
	v_mov_b32_e32 v15, 0                                       // 00000000001C: 7E1E0280 
	s_lshl_b32 s11, s10, 5                                     // 000000000020: 840B850A 
	s_lshl_b32 s10, s0, 5                                      // 000000000024: 840A8500 
	v_and_b32_e32 v128, 15, v129                               // 000000000028: 3701028F 
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000002C: BF870002 
	v_dual_mov_b32 v14, v15 :: v_dual_lshlrev_b32 v41, 1, v129 // 000000000030: CA22010F 0E290281 
	v_mov_b32_e32 v13, v15                                     // 000000000038: 7E1A030F 
	v_mov_b32_e32 v12, v15                                     // 00000000003C: 7E18030F 
	v_mov_b32_e32 v11, v15                                     // 000000000040: 7E16030F 
	v_mov_b32_e32 v10, v15                                     // 000000000044: 7E14030F 
	v_mov_b32_e32 v9, v15                                      // 000000000048: 7E12030F 
	v_mov_b32_e32 v8, v15                                      // 00000000004C: 7E10030F 
	v_mov_b32_e32 v23, v15                                     // 000000000050: 7E2E030F 
	v_mov_b32_e32 v22, v15                                     // 000000000054: 7E2C030F 
	v_mov_b32_e32 v21, v15                                     // 000000000058: 7E2A030F 
	v_mov_b32_e32 v20, v15                                     // 00000000005C: 7E28030F 
	v_mov_b32_e32 v19, v15                                     // 000000000060: 7E26030F 
	v_mov_b32_e32 v18, v15                                     // 000000000064: 7E24030F 
	v_mov_b32_e32 v17, v15                                     // 000000000068: 7E22030F 
	v_mov_b32_e32 v16, v15                                     // 00000000006C: 7E20030F 
	v_mov_b32_e32 v49, v15                                     // 000000000070: 7E62030F 
	v_mov_b32_e32 v48, v15                                     // 000000000074: 7E60030F 
	v_mov_b32_e32 v47, v15                                     // 000000000078: 7E5E030F 
	v_mov_b32_e32 v46, v15                                     // 00000000007C: 7E5C030F 
	v_mov_b32_e32 v45, v15                                     // 000000000080: 7E5A030F 
	v_mov_b32_e32 v44, v15                                     // 000000000084: 7E58030F 
	v_mov_b32_e32 v43, v15                                     // 000000000088: 7E56030F 
	v_mov_b32_e32 v42, v15                                     // 00000000008C: 7E54030F 
	v_mov_b32_e32 v255, v15                                    // 000000000090: 7FFE030F 
	v_mov_b32_e32 v254, v15                                    // 000000000094: 7FFC030F 
	v_mov_b32_e32 v253, v15                                    // 000000000098: 7FFA030F 
	v_mov_b32_e32 v252, v15                                    // 00000000009C: 7FF8030F 
	v_mov_b32_e32 v251, v15                                    // 0000000000A0: 7FF6030F 
	v_mov_b32_e32 v250, v15                                    // 0000000000A4: 7FF4030F 
	v_mov_b32_e32 v249, v15                                    // 0000000000A8: 7FF2030F 
	v_mov_b32_e32 v248, v15                                    // 0000000000AC: 7FF0030F 
	v_mov_b32_e32 v247, v15                                    // 0000000000B0: 7FEE030F 
	v_mov_b32_e32 v246, v15                                    // 0000000000B4: 7FEC030F 
	v_mov_b32_e32 v245, v15                                    // 0000000000B8: 7FEA030F 
	v_mov_b32_e32 v244, v15                                    // 0000000000BC: 7FE8030F 
	v_mov_b32_e32 v243, v15                                    // 0000000000C0: 7FE6030F 
	v_mov_b32_e32 v242, v15                                    // 0000000000C4: 7FE4030F 
	v_mov_b32_e32 v241, v15                                    // 0000000000C8: 7FE2030F 
	v_mov_b32_e32 v240, v15                                    // 0000000000CC: 7FE0030F 
	v_mov_b32_e32 v88, v15                                     // 0000000000D0: 7EB0030F 
	v_mov_b32_e32 v87, v15                                     // 0000000000D4: 7EAE030F 
	v_mov_b32_e32 v86, v15                                     // 0000000000D8: 7EAC030F 
	v_mov_b32_e32 v85, v15                                     // 0000000000DC: 7EAA030F 
	v_mov_b32_e32 v84, v15                                     // 0000000000E0: 7EA8030F 
	v_mov_b32_e32 v83, v15                                     // 0000000000E4: 7EA6030F 
	v_mov_b32_e32 v82, v15                                     // 0000000000E8: 7EA4030F 
	v_mov_b32_e32 v81, v15                                     // 0000000000EC: 7EA2030F 
	v_mov_b32_e32 v96, v15                                     // 0000000000F0: 7EC0030F 
	v_mov_b32_e32 v95, v15                                     // 0000000000F4: 7EBE030F 
	v_mov_b32_e32 v94, v15                                     // 0000000000F8: 7EBC030F 
	v_mov_b32_e32 v93, v15                                     // 0000000000FC: 7EBA030F 
	v_mov_b32_e32 v92, v15                                     // 000000000100: 7EB8030F 
	v_mov_b32_e32 v91, v15                                     // 000000000104: 7EB6030F 
	v_mov_b32_e32 v90, v15                                     // 000000000108: 7EB4030F 
	v_mov_b32_e32 v89, v15                                     // 00000000010C: 7EB2030F 
	v_mov_b32_e32 v104, v15                                    // 000000000110: 7ED0030F 
	v_mov_b32_e32 v103, v15                                    // 000000000114: 7ECE030F 
	v_mov_b32_e32 v102, v15                                    // 000000000118: 7ECC030F 
	v_mov_b32_e32 v101, v15                                    // 00000000011C: 7ECA030F 
	v_mov_b32_e32 v100, v15                                    // 000000000120: 7EC8030F 
	v_mov_b32_e32 v99, v15                                     // 000000000124: 7EC6030F 
	v_mov_b32_e32 v98, v15                                     // 000000000128: 7EC4030F 
	v_mov_b32_e32 v97, v15                                     // 00000000012C: 7EC2030F 
	v_mov_b32_e32 v7, v15                                      // 000000000130: 7E0E030F 
	v_mov_b32_e32 v6, v15                                      // 000000000134: 7E0C030F 
	v_mov_b32_e32 v5, v15                                      // 000000000138: 7E0A030F 
	v_mov_b32_e32 v4, v15                                      // 00000000013C: 7E08030F 
	v_mov_b32_e32 v3, v15                                      // 000000000140: 7E06030F 
	v_mov_b32_e32 v2, v15                                      // 000000000144: 7E04030F 
	v_mov_b32_e32 v1, v15                                      // 000000000148: 7E02030F 
	v_mov_b32_e32 v0, v15                                      // 00000000014C: 7E00030F 
	v_mov_b32_e32 v215, v15                                    // 000000000150: 7FAE030F 
	v_mov_b32_e32 v214, v15                                    // 000000000154: 7FAC030F 
	v_mov_b32_e32 v213, v15                                    // 000000000158: 7FAA030F 
	v_mov_b32_e32 v212, v15                                    // 00000000015C: 7FA8030F 
	v_mov_b32_e32 v211, v15                                    // 000000000160: 7FA6030F 
	v_mov_b32_e32 v210, v15                                    // 000000000164: 7FA4030F 
	v_mov_b32_e32 v209, v15                                    // 000000000168: 7FA2030F 
	v_mov_b32_e32 v208, v15                                    // 00000000016C: 7FA0030F 
	v_mov_b32_e32 v57, v15                                     // 000000000170: 7E72030F 
	v_mov_b32_e32 v56, v15                                     // 000000000174: 7E70030F 
	v_mov_b32_e32 v55, v15                                     // 000000000178: 7E6E030F 
	v_mov_b32_e32 v54, v15                                     // 00000000017C: 7E6C030F 
	v_mov_b32_e32 v53, v15                                     // 000000000180: 7E6A030F 
	v_mov_b32_e32 v52, v15                                     // 000000000184: 7E68030F 
	v_mov_b32_e32 v51, v15                                     // 000000000188: 7E66030F 
	v_mov_b32_e32 v50, v15                                     // 00000000018C: 7E64030F 
	v_mov_b32_e32 v65, v15                                     // 000000000190: 7E82030F 
	v_mov_b32_e32 v64, v15                                     // 000000000194: 7E80030F 
	v_mov_b32_e32 v63, v15                                     // 000000000198: 7E7E030F 
	v_mov_b32_e32 v62, v15                                     // 00000000019C: 7E7C030F 
	v_mov_b32_e32 v61, v15                                     // 0000000001A0: 7E7A030F 
	v_mov_b32_e32 v60, v15                                     // 0000000001A4: 7E78030F 
	v_mov_b32_e32 v59, v15                                     // 0000000001A8: 7E76030F 
	v_mov_b32_e32 v58, v15                                     // 0000000001AC: 7E74030F 
	v_mov_b32_e32 v112, v15                                    // 0000000001B0: 7EE0030F 
	v_mov_b32_e32 v111, v15                                    // 0000000001B4: 7EDE030F 
	v_mov_b32_e32 v110, v15                                    // 0000000001B8: 7EDC030F 
	v_mov_b32_e32 v109, v15                                    // 0000000001BC: 7EDA030F 
	v_mov_b32_e32 v108, v15                                    // 0000000001C0: 7ED8030F 
	v_mov_b32_e32 v107, v15                                    // 0000000001C4: 7ED6030F 
	v_mov_b32_e32 v106, v15                                    // 0000000001C8: 7ED4030F 
	v_mov_b32_e32 v105, v15                                    // 0000000001CC: 7ED2030F 
	v_mov_b32_e32 v120, v15                                    // 0000000001D0: 7EF0030F 
	v_mov_b32_e32 v119, v15                                    // 0000000001D4: 7EEE030F 
	v_mov_b32_e32 v118, v15                                    // 0000000001D8: 7EEC030F 
	v_mov_b32_e32 v117, v15                                    // 0000000001DC: 7EEA030F 
	v_mov_b32_e32 v116, v15                                    // 0000000001E0: 7EE8030F 
	v_mov_b32_e32 v115, v15                                    // 0000000001E4: 7EE6030F 
	v_mov_b32_e32 v114, v15                                    // 0000000001E8: 7EE4030F 
	v_mov_b32_e32 v113, v15                                    // 0000000001EC: 7EE2030F 
	v_mov_b32_e32 v175, v15                                    // 0000000001F0: 7F5E030F 
	v_mov_b32_e32 v174, v15                                    // 0000000001F4: 7F5C030F 
	v_mov_b32_e32 v173, v15                                    // 0000000001F8: 7F5A030F 
	v_mov_b32_e32 v172, v15                                    // 0000000001FC: 7F58030F 
	v_mov_b32_e32 v171, v15                                    // 000000000200: 7F56030F 
	v_mov_b32_e32 v170, v15                                    // 000000000204: 7F54030F 
	v_mov_b32_e32 v169, v15                                    // 000000000208: 7F52030F 
	v_mov_b32_e32 v168, v15                                    // 00000000020C: 7F50030F 
	v_mov_b32_e32 v167, v15                                    // 000000000210: 7F4E030F 
	v_mov_b32_e32 v166, v15                                    // 000000000214: 7F4C030F 
	v_mov_b32_e32 v165, v15                                    // 000000000218: 7F4A030F 
	v_mov_b32_e32 v164, v15                                    // 00000000021C: 7F48030F 
	v_mov_b32_e32 v163, v15                                    // 000000000220: 7F46030F 
	v_mov_b32_e32 v162, v15                                    // 000000000224: 7F44030F 
	v_mov_b32_e32 v161, v15                                    // 000000000228: 7F42030F 
	v_mov_b32_e32 v160, v15                                    // 00000000022C: 7F40030F 
	s_cmp_eq_u32 s3, 0                                         // 000000000230: BF068003 
	s_mov_b32 s24, 0                                           // 000000000234: BE980080 
	s_clause 0x1                                               // 000000000238: BF850001 
	scratch_store_b128 off, v[0:3], off                        // 00000000023C: DC750000 007C0000 
	scratch_store_b128 off, v[4:7], off offset:16              // 000000000244: DC750010 007C0400 
	s_cbranch_scc1 5559                                        // 00000000024C: BFA215B7  <_amdgpu_cs_main+0x592C>
	v_mov_b32_e32 v8, 0                                        // 000000000250: 7E100280 
	v_lshrrev_b32_e32 v0, 3, v129                              // 000000000254: 32010283 
	s_load_b256 s[12:19], s[8:9], null                         // 000000000258: F40C0304 F8000000 
	s_bitcmp1_b32 s7, 0                                        // 000000000260: BF0D8007 
	s_mov_b32 s28, 0                                           // 000000000264: BE9C0080 
	v_mov_b32_e32 v1, v8                                       // 000000000268: 7E020308 
	scratch_store_b32 off, v0, off offset:580                  // 00000000026C: DC690244 007C0000 
	v_mov_b32_e32 v0, v8                                       // 000000000274: 7E000308 
	v_mov_b32_e32 v2, v8                                       // 000000000278: 7E040308 
	v_mov_b32_e32 v3, v8                                       // 00000000027C: 7E060308 
	v_mov_b32_e32 v4, v8                                       // 000000000280: 7E080308 
	v_mov_b32_e32 v5, v8                                       // 000000000284: 7E0A0308 
	v_mov_b32_e32 v6, v8                                       // 000000000288: 7E0C0308 
	v_mov_b32_e32 v7, v8                                       // 00000000028C: 7E0E0308 
	s_clause 0x1                                               // 000000000290: BF850001 
	scratch_store_b128 off, v[0:3], off offset:32              // 000000000294: DC750020 007C0000 
	scratch_store_b128 off, v[4:7], off offset:48              // 00000000029C: DC750030 007C0400 
	s_cselect_b32 s25, -1, 0                                   // 0000000002A4: 981980C1 
	s_bitcmp1_b32 s7, 1                                        // 0000000002A8: BF0D8107 
	s_cselect_b32 s9, -1, 0                                    // 0000000002AC: 980980C1 
	s_bitcmp0_b32 s4, 0                                        // 0000000002B0: BF0C8004 
	s_cselect_b32 s26, -1, 0                                   // 0000000002B4: 981A80C1 
	s_bitcmp0_b32 s5, 0                                        // 0000000002B8: BF0C8005 
	s_cselect_b32 s27, -1, 0                                   // 0000000002BC: 981B80C1 
	v_dual_mov_b32 v144, v7 :: v_dual_mov_b32 v141, v4         // 0000000002C0: CA100107 908C0104 
	v_mov_b32_e32 v143, v6                                     // 0000000002C8: 7F1E0306 
	v_dual_mov_b32 v9, v8 :: v_dual_mov_b32 v142, v5           // 0000000002CC: CA100108 098E0105 
	v_dual_mov_b32 v10, v8 :: v_dual_mov_b32 v139, v2          // 0000000002D4: CA100108 0A8A0102 
	v_dual_mov_b32 v11, v8 :: v_dual_mov_b32 v140, v3          // 0000000002DC: CA100108 0B8C0103 
	v_dual_mov_b32 v13, v8 :: v_dual_mov_b32 v138, v1          // 0000000002E4: CA100108 0D8A0101 
	v_mov_b32_e32 v137, v0                                     // 0000000002EC: 7F120300 
	s_clause 0xf                                               // 0000000002F0: BF85000F 
	scratch_store_b128 off, v[0:3], off offset:128             // 0000000002F4: DC750080 007C0000 
	scratch_store_b128 off, v[4:7], off offset:144             // 0000000002FC: DC750090 007C0400 
	scratch_store_b128 off, v[0:3], off offset:224             // 000000000304: DC7500E0 007C0000 
	scratch_store_b128 off, v[4:7], off offset:240             // 00000000030C: DC7500F0 007C0400 
	scratch_store_b128 off, v[0:3], off offset:256             // 000000000314: DC750100 007C0000 
	scratch_store_b128 off, v[4:7], off offset:272             // 00000000031C: DC750110 007C0400 
	scratch_store_b128 off, v[0:3], off offset:160             // 000000000324: DC7500A0 007C0000 
	scratch_store_b128 off, v[4:7], off offset:176             // 00000000032C: DC7500B0 007C0400 
	scratch_store_b128 off, v[0:3], off offset:192             // 000000000334: DC7500C0 007C0000 
	scratch_store_b128 off, v[4:7], off offset:208             // 00000000033C: DC7500D0 007C0400 
	scratch_store_b128 off, v[0:3], off                        // 000000000344: DC750000 007C0000 
	scratch_store_b128 off, v[4:7], off offset:16              // 00000000034C: DC750010 007C0400 
	scratch_store_b128 off, v[0:3], off offset:64              // 000000000354: DC750040 007C0000 
	scratch_store_b128 off, v[4:7], off offset:80              // 00000000035C: DC750050 007C0400 
	scratch_store_b128 off, v[0:3], off offset:96              // 000000000364: DC750060 007C0000 
	scratch_store_b128 off, v[4:7], off offset:112             // 00000000036C: DC750070 007C0400 
	v_mov_b32_e32 v12, v8                                      // 000000000374: 7E180308 
	v_mov_b32_e32 v14, v8                                      // 000000000378: 7E1C0308 
	v_mov_b32_e32 v15, v8                                      // 00000000037C: 7E1E0308 
	v_mov_b32_e32 v232, v8                                     // 000000000380: 7FD00308 
	v_mov_b32_e32 v233, v8                                     // 000000000384: 7FD20308 
	v_mov_b32_e32 v234, v8                                     // 000000000388: 7FD40308 
	v_mov_b32_e32 v235, v8                                     // 00000000038C: 7FD60308 
	v_mov_b32_e32 v236, v8                                     // 000000000390: 7FD80308 
	v_mov_b32_e32 v237, v8                                     // 000000000394: 7FDA0308 
	v_mov_b32_e32 v238, v8                                     // 000000000398: 7FDC0308 
	v_mov_b32_e32 v239, v8                                     // 00000000039C: 7FDE0308 
	v_mov_b32_e32 v216, v8                                     // 0000000003A0: 7FB00308 
	v_mov_b32_e32 v217, v8                                     // 0000000003A4: 7FB20308 
	v_mov_b32_e32 v218, v8                                     // 0000000003A8: 7FB40308 
	v_mov_b32_e32 v219, v8                                     // 0000000003AC: 7FB60308 
	v_mov_b32_e32 v220, v8                                     // 0000000003B0: 7FB80308 
	v_mov_b32_e32 v221, v8                                     // 0000000003B4: 7FBA0308 
	v_mov_b32_e32 v222, v8                                     // 0000000003B8: 7FBC0308 
	v_mov_b32_e32 v223, v8                                     // 0000000003BC: 7FBE0308 
	v_mov_b32_e32 v184, v8                                     // 0000000003C0: 7F700308 
	v_mov_b32_e32 v185, v8                                     // 0000000003C4: 7F720308 
	v_mov_b32_e32 v186, v8                                     // 0000000003C8: 7F740308 
	v_mov_b32_e32 v187, v8                                     // 0000000003CC: 7F760308 
	v_mov_b32_e32 v188, v8                                     // 0000000003D0: 7F780308 
	v_mov_b32_e32 v189, v8                                     // 0000000003D4: 7F7A0308 
	v_mov_b32_e32 v190, v8                                     // 0000000003D8: 7F7C0308 
	v_mov_b32_e32 v191, v8                                     // 0000000003DC: 7F7E0308 
	v_mov_b32_e32 v32, v8                                      // 0000000003E0: 7E400308 
	v_mov_b32_e32 v33, v8                                      // 0000000003E4: 7E420308 
	v_mov_b32_e32 v34, v8                                      // 0000000003E8: 7E440308 
	v_mov_b32_e32 v35, v8                                      // 0000000003EC: 7E460308 
	v_mov_b32_e32 v36, v8                                      // 0000000003F0: 7E480308 
	v_mov_b32_e32 v37, v8                                      // 0000000003F4: 7E4A0308 
	v_mov_b32_e32 v38, v8                                      // 0000000003F8: 7E4C0308 
	v_mov_b32_e32 v39, v8                                      // 0000000003FC: 7E4E0308 
	s_clause 0x1                                               // 000000000400: BF850001 
	scratch_store_b128 off, v[0:3], off offset:288             // 000000000404: DC750120 007C0000 
	scratch_store_b128 off, v[4:7], off offset:304             // 00000000040C: DC750130 007C0400 
	s_branch 486                                               // 000000000414: BFA001E6  <_amdgpu_cs_main+0xBB0>
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:548                  // 000000000420: DC510224 197C0000 
 <unknown>
	scratch_load_b32 v1, off, off offset:532                   // 00000000042C: DC510214 017C0000 
 <unknown>
	scratch_load_b32 v28, off, off offset:536                  // 000000000438: DC510218 1C7C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_load_b32 v32, off, off offset:552                  // 000000000460: DC510228 207C0000 
	s_waitcnt vmcnt(6)                                         // 000000000468: BF891BF7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(4)                                         // 0000000005A0: BF8913F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_mbcnt_lo_u32_b32 v129, -1, 0                             // 000000000698: D71F0081 000100C1 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v236, v62 :: v_dual_mov_b32 v237, v63       // 0000000006B8: CA10013E ECEC013F 
	v_dual_mov_b32 v238, v64 :: v_dual_mov_b32 v239, v65       // 0000000006C0: CA100140 EEEE0141 
	v_dual_mov_b32 v216, v105 :: v_dual_mov_b32 v217, v106     // 0000000006C8: CA100169 D8D8016A 
	v_dual_mov_b32 v218, v107 :: v_dual_mov_b32 v219, v108     // 0000000006D0: CA10016B DADA016C 
	v_dual_mov_b32 v220, v109 :: v_dual_mov_b32 v221, v110     // 0000000006D8: CA10016D DCDC016E 
	v_dual_mov_b32 v222, v111 :: v_dual_mov_b32 v223, v112     // 0000000006E0: CA10016F DEDE0170 
	v_dual_mov_b32 v184, v113 :: v_dual_mov_b32 v185, v114     // 0000000006E8: CA100171 B8B80172 
	v_dual_mov_b32 v186, v115 :: v_dual_mov_b32 v187, v116     // 0000000006F0: CA100173 BABA0174 
	v_dual_mov_b32 v188, v117 :: v_dual_mov_b32 v189, v118     // 0000000006F8: CA100175 BCBC0176 
	v_dual_mov_b32 v190, v119 :: v_dual_mov_b32 v191, v120     // 000000000700: CA100177 BEBE0178 
	v_dual_mov_b32 v34, v162 :: v_dual_mov_b32 v35, v163       // 000000000708: CA1001A2 222201A3 
	v_dual_mov_b32 v36, v164 :: v_dual_mov_b32 v37, v165       // 000000000710: CA1001A4 242401A5 
	v_dual_mov_b32 v38, v166 :: v_dual_mov_b32 v39, v167       // 000000000718: CA1001A6 262601A7 
	v_and_b32_e32 v128, 15, v129                               // 000000000720: 3701028F 
	s_add_i32 s24, s24, 16                                     // 000000000724: 81189018 
	s_add_i32 s28, s28, 1                                      // 000000000728: 811C811C 
	s_cmp_lt_u32 s24, s3                                       // 00000000072C: BF0A0318 
	v_mov_b32_e32 v33, v161                                    // 000000000730: 7E4203A1 
	s_waitcnt vmcnt(3)                                         // 000000000734: BF890FF7 
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:476                  // 000000000740: DC5101DC 197C0000 
	s_waitcnt vmcnt(3)                                         // 000000000748: BF890FF7 
 <unknown>
 <unknown>
	scratch_load_b32 v1, off, off offset:520                   // 000000000754: DC510208 017C0000 
	s_waitcnt vmcnt(3)                                         // 00000000075C: BF890FF7 
 <unknown>
	scratch_load_b32 v28, off, off offset:516                  // 000000000764: DC510204 1C7C0000 
	s_waitcnt vmcnt(2)                                         // 00000000076C: BF890BF7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 000000000778: BF8907F7 
 <unknown>
 <unknown>
	s_clause 0x2                                               // 000000000784: BF850002 
	scratch_load_b128 v[0:3], off, off                         // 000000000788: DC5D0000 007C0000 
	scratch_load_b128 v[4:7], off, off offset:16               // 000000000790: DC5D0010 047C0000 
	scratch_load_b32 v24, off, off offset:540                  // 000000000798: DC51021C 187C0000 
	s_waitcnt vmcnt(1)                                         // 0000000007A0: BF8907F7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 0000000007AC: BF8903F7 
 <unknown>
	s_clause 0x1                                               // 0000000007B4: BF850001 
	scratch_load_b32 v24, off, off offset:528                  // 0000000007B8: DC510210 187C0000 
	scratch_load_b32 v25, off, off offset:472                  // 0000000007C0: DC5101D8 197C0000 
 <unknown>
	scratch_load_b32 v28, off, off offset:576                  // 0000000007CC: DC510240 1C7C0000 
	s_waitcnt vmcnt(2)                                         // 0000000007D4: BF890BF7 
 <unknown>
	scratch_load_b32 v24, off, off offset:524                  // 0000000007DC: DC51020C 187C0000 
	s_waitcnt vmcnt(2)                                         // 0000000007E4: BF890BF7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 0000000007F0: BF8907F7 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 0000000007FC: BF850001 
	scratch_load_b32 v28, off, off offset:568                  // 000000000800: DC510238 1C7C0000 
	scratch_load_b32 v25, off, off offset:468                  // 000000000808: DC5101D4 197C0000 
	s_waitcnt vmcnt(2)                                         // 000000000810: BF890BF7 
 <unknown>
	scratch_load_b32 v24, off, off offset:544                  // 000000000818: DC510220 187C0000 
	s_waitcnt vmcnt(2)                                         // 000000000820: BF890BF7 
 <unknown>
 <unknown>
	scratch_load_b32 v28, off, off offset:496                  // 00000000082C: DC5101F0 1C7C0000 
	s_waitcnt vmcnt(1)                                         // 000000000834: BF8907F7 
 <unknown>
	scratch_load_b32 v24, off, off offset:512                  // 00000000083C: DC510200 187C0000 
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:464                  // 00000000084C: DC5101D0 197C0000 
	s_waitcnt vmcnt(1)                                         // 000000000854: BF8907F7 
 <unknown>
	scratch_load_b32 v24, off, off offset:508                  // 00000000085C: DC5101FC 187C0000 
	s_waitcnt vmcnt(1)                                         // 000000000864: BF8907F7 
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:460                  // 000000000870: DC5101CC 197C0000 
	s_waitcnt vmcnt(1)                                         // 000000000878: BF8907F7 
 <unknown>
	scratch_load_b32 v24, off, off offset:504                  // 000000000880: DC5101F8 187C0000 
	s_waitcnt vmcnt(1)                                         // 000000000888: BF8907F7 
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:456                  // 000000000894: DC5101C8 197C0000 
 <unknown>
	scratch_load_b32 v28, off, off offset:480                  // 0000000008A0: DC5101E0 1C7C0000 
 <unknown>
	scratch_load_b32 v32, off, off offset:384                  // 0000000008AC: DC510180 207C0000 
	s_waitcnt vmcnt(3)                                         // 0000000008B4: BF890FF7 
 <unknown>
	s_clause 0x1                                               // 0000000008BC: BF850001 
	scratch_store_b128 off, v[0:3], off                        // 0000000008C0: DC750000 007C0000 
	scratch_store_b128 off, v[4:7], off offset:16              // 0000000008C8: DC750010 007C0400 
	s_waitcnt vmcnt(2)                                         // 0000000008D0: BF890BF7 
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:452                  // 0000000008DC: DC5101C4 197C0000 
	s_waitcnt vmcnt(2)                                         // 0000000008E4: BF890BF7 
 <unknown>
	scratch_load_b32 v28, off, off offset:572                  // 0000000008EC: DC51023C 1C7C0000 
	v_dual_mov_b32 v0, v16 :: v_dual_mov_b32 v1, v17           // 0000000008F4: CA100110 00000111 
	v_dual_mov_b32 v2, v18 :: v_dual_mov_b32 v3, v19           // 0000000008FC: CA100112 02020113 
	v_dual_mov_b32 v4, v20 :: v_dual_mov_b32 v5, v21           // 000000000904: CA100114 04040115 
	v_dual_mov_b32 v6, v22 :: v_dual_mov_b32 v7, v23           // 00000000090C: CA100116 06060117 
	s_clause 0x1                                               // 000000000914: BF850001 
	scratch_store_b128 off, v[0:3], off offset:32              // 000000000918: DC750020 007C0000 
	scratch_store_b128 off, v[4:7], off offset:48              // 000000000920: DC750030 007C0400 
	v_dual_mov_b32 v6, v48 :: v_dual_mov_b32 v7, v49           // 000000000928: CA100130 06060131 
	v_dual_mov_b32 v4, v46 :: v_dual_mov_b32 v5, v47           // 000000000930: CA10012E 0404012F 
	s_waitcnt vmcnt(2)                                         // 000000000938: BF890BF7 
 <unknown>
	v_mov_b32_e32 v32, v160                                    // 000000000940: 7E4003A0 
	s_waitcnt vmcnt(1)                                         // 000000000944: BF8907F7 
 <unknown>
 <unknown>
	scratch_load_b32 v25, off, off offset:560                  // 000000000950: DC510230 197C0000 
	s_waitcnt vmcnt(0)                                         // 000000000958: BF8903F7 
 <unknown>
	scratch_load_b32 v25, off, off offset:564                  // 000000000960: DC510234 197C0000 
	s_waitcnt vmcnt(0)                                         // 000000000968: BF8903F7 
 <unknown>
	scratch_load_b32 v25, off, off offset:556                  // 000000000970: DC51022C 197C0000 
	v_dual_mov_b32 v232, v58 :: v_dual_mov_b32 v233, v59       // 000000000978: CA10013A E8E8013B 
	s_waitcnt vmcnt(0)                                         // 000000000980: BF8903F7 
 <unknown>
	scratch_load_b32 v25, off, off offset:500                  // 000000000988: DC5101F4 197C0000 
	v_dual_mov_b32 v234, v60 :: v_dual_mov_b32 v235, v61       // 000000000990: CA10013C EAEA013D 
	s_waitcnt vmcnt(0)                                         // 000000000998: BF8903F7 
 <unknown>
	scratch_load_b32 v25, off, off offset:492                  // 0000000009A0: DC5101EC 197C0000 
	s_waitcnt vmcnt(0)                                         // 0000000009A8: BF8903F7 
 <unknown>
	scratch_load_b32 v25, off, off offset:488                  // 0000000009B0: DC5101E8 197C0000 
	v_dual_mov_b32 v0, v42 :: v_dual_mov_b32 v1, v43           // 0000000009B8: CA10012A 0000012B 
	s_waitcnt vmcnt(0)                                         // 0000000009C0: BF8903F7 
 <unknown>
	scratch_load_b32 v25, off, off offset:484                  // 0000000009C8: DC5101E4 197C0000 
 <unknown>
	s_waitcnt vmcnt(0)                                         // 0000000009D4: BF8903F7 
 <unknown>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 0000000009DC: BF870241 
	v_dual_mov_b32 v2, v44 :: v_dual_mov_b32 v3, v45           // 0000000009E0: CA10012C 0202012D 
	v_dual_mov_b32 v144, v7 :: v_dual_mov_b32 v143, v6         // 0000000009E8: CA100107 908E0106 
	v_dual_mov_b32 v142, v5 :: v_dual_mov_b32 v141, v4         // 0000000009F0: CA100105 8E8C0104 
	v_dual_mov_b32 v138, v1 :: v_dual_mov_b32 v137, v0         // 0000000009F8: CA100101 8A880100 
	v_dual_mov_b32 v140, v3 :: v_dual_mov_b32 v139, v2         // 000000000A00: CA100103 8C8A0102 
	v_dual_mov_b32 v0, v248 :: v_dual_mov_b32 v1, v249         // 000000000A08: CA1001F8 000001F9 
	v_dual_mov_b32 v2, v250 :: v_dual_mov_b32 v3, v251         // 000000000A10: CA1001FA 020201FB 
	v_dual_mov_b32 v4, v252 :: v_dual_mov_b32 v5, v253         // 000000000A18: CA1001FC 040401FD 
	v_dual_mov_b32 v6, v254 :: v_dual_mov_b32 v7, v255         // 000000000A20: CA1001FE 060601FF 
	s_clause 0x1                                               // 000000000A28: BF850001 
	scratch_store_b128 off, v[0:3], off offset:128             // 000000000A2C: DC750080 007C0000 
	scratch_store_b128 off, v[4:7], off offset:144             // 000000000A34: DC750090 007C0400 
	v_dual_mov_b32 v0, v240 :: v_dual_mov_b32 v1, v241         // 000000000A3C: CA1001F0 000001F1 
	v_dual_mov_b32 v2, v242 :: v_dual_mov_b32 v3, v243         // 000000000A44: CA1001F2 020201F3 
	v_dual_mov_b32 v4, v244 :: v_dual_mov_b32 v5, v245         // 000000000A4C: CA1001F4 040401F5 
	v_dual_mov_b32 v6, v246 :: v_dual_mov_b32 v7, v247         // 000000000A54: CA1001F6 060601F7 
	s_clause 0x1                                               // 000000000A5C: BF850001 
	scratch_store_b128 off, v[0:3], off offset:224             // 000000000A60: DC7500E0 007C0000 
	scratch_store_b128 off, v[4:7], off offset:240             // 000000000A68: DC7500F0 007C0400 
	v_dual_mov_b32 v0, v81 :: v_dual_mov_b32 v1, v82           // 000000000A70: CA100151 00000152 
	v_dual_mov_b32 v2, v83 :: v_dual_mov_b32 v3, v84           // 000000000A78: CA100153 02020154 
	v_dual_mov_b32 v4, v85 :: v_dual_mov_b32 v5, v86           // 000000000A80: CA100155 04040156 
	v_dual_mov_b32 v6, v87 :: v_dual_mov_b32 v7, v88           // 000000000A88: CA100157 06060158 
	s_clause 0x1                                               // 000000000A90: BF850001 
	scratch_store_b128 off, v[0:3], off offset:256             // 000000000A94: DC750100 007C0000 
	scratch_store_b128 off, v[4:7], off offset:272             // 000000000A9C: DC750110 007C0400 
	v_dual_mov_b32 v0, v89 :: v_dual_mov_b32 v1, v90           // 000000000AA4: CA100159 0000015A 
	v_dual_mov_b32 v2, v91 :: v_dual_mov_b32 v3, v92           // 000000000AAC: CA10015B 0202015C 
	v_dual_mov_b32 v4, v93 :: v_dual_mov_b32 v5, v94           // 000000000AB4: CA10015D 0404015E 
	v_dual_mov_b32 v6, v95 :: v_dual_mov_b32 v7, v96           // 000000000ABC: CA10015F 06060160 
	s_clause 0x1                                               // 000000000AC4: BF850001 
	scratch_store_b128 off, v[0:3], off offset:160             // 000000000AC8: DC7500A0 007C0000 
	scratch_store_b128 off, v[4:7], off offset:176             // 000000000AD0: DC7500B0 007C0400 
	v_dual_mov_b32 v0, v97 :: v_dual_mov_b32 v1, v98           // 000000000AD8: CA100161 00000162 
	v_dual_mov_b32 v2, v99 :: v_dual_mov_b32 v3, v100          // 000000000AE0: CA100163 02020164 
	v_dual_mov_b32 v4, v101 :: v_dual_mov_b32 v5, v102         // 000000000AE8: CA100165 04040166 
	v_dual_mov_b32 v6, v103 :: v_dual_mov_b32 v7, v104         // 000000000AF0: CA100167 06060168 
	s_clause 0x1                                               // 000000000AF8: BF850001 
	scratch_store_b128 off, v[0:3], off offset:192             // 000000000AFC: DC7500C0 007C0000 
	scratch_store_b128 off, v[4:7], off offset:208             // 000000000B04: DC7500D0 007C0400 
	v_dual_mov_b32 v0, v208 :: v_dual_mov_b32 v1, v209         // 000000000B0C: CA1001D0 000001D1 
	v_dual_mov_b32 v2, v210 :: v_dual_mov_b32 v3, v211         // 000000000B14: CA1001D2 020201D3 
	v_dual_mov_b32 v4, v212 :: v_dual_mov_b32 v5, v213         // 000000000B1C: CA1001D4 040401D5 
	v_dual_mov_b32 v6, v214 :: v_dual_mov_b32 v7, v215         // 000000000B24: CA1001D6 060601D7 
	s_clause 0x1                                               // 000000000B2C: BF850001 
	scratch_store_b128 off, v[0:3], off offset:64              // 000000000B30: DC750040 007C0000 
	scratch_store_b128 off, v[4:7], off offset:80              // 000000000B38: DC750050 007C0400 
	v_dual_mov_b32 v0, v50 :: v_dual_mov_b32 v1, v51           // 000000000B40: CA100132 00000133 
	v_dual_mov_b32 v2, v52 :: v_dual_mov_b32 v3, v53           // 000000000B48: CA100134 02020135 
	v_dual_mov_b32 v4, v54 :: v_dual_mov_b32 v5, v55           // 000000000B50: CA100136 04040137 
	v_dual_mov_b32 v6, v56 :: v_dual_mov_b32 v7, v57           // 000000000B58: CA100138 06060139 
	s_clause 0x1                                               // 000000000B60: BF850001 
	scratch_store_b128 off, v[0:3], off offset:96              // 000000000B64: DC750060 007C0000 
	scratch_store_b128 off, v[4:7], off offset:112             // 000000000B6C: DC750070 007C0400 
	v_dual_mov_b32 v0, v168 :: v_dual_mov_b32 v1, v169         // 000000000B74: CA1001A8 000001A9 
	v_dual_mov_b32 v2, v170 :: v_dual_mov_b32 v3, v171         // 000000000B7C: CA1001AA 020201AB 
	v_dual_mov_b32 v4, v172 :: v_dual_mov_b32 v5, v173         // 000000000B84: CA1001AC 040401AD 
	v_dual_mov_b32 v6, v174 :: v_dual_mov_b32 v7, v175         // 000000000B8C: CA1001AE 060601AF 
	s_clause 0x1                                               // 000000000B94: BF850001 
	scratch_store_b128 off, v[0:3], off offset:288             // 000000000B98: DC750120 007C0000 
	scratch_store_b128 off, v[4:7], off offset:304             // 000000000BA0: DC750130 007C0400 
	s_waitcnt_vscnt null, 0x0                                  // 000000000BA8: BC7C0000 
	s_cbranch_scc0 4958                                        // 000000000BAC: BFA1135E  <_amdgpu_cs_main+0x5928>
	v_cmp_gt_u32_e64 s0, 0x100, v129                           // 000000000BB0: D44C0000 000302FF 00000100 
	s_and_not1_b32 vcc_lo, exec_lo, s25                        // 000000000BBC: 916A197E 
	s_mov_b32 s7, -1                                           // 000000000BC0: BE8700C1 
	s_cbranch_vccnz 95                                         // 000000000BC4: BFA4005F  <_amdgpu_cs_main+0xD44>
	s_and_saveexec_b32 s29, s0                                 // 000000000BC8: BE9D2000 
	s_cbranch_execz 91                                         // 000000000BCC: BFA5005B  <_amdgpu_cs_main+0xD3C>
	v_or_b32_e32 v17, s24, v128                                // 000000000BD0: 38230018 
	s_mov_b32 s31, 0                                           // 000000000BD4: BE9F0080 
	v_mov_b32_e32 v19, v129                                    // 000000000BD8: 7E260381 
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000000BDC: BF8700D2 
	v_mul_lo_u32 v16, v17, s4                                  // 000000000BE0: D72C0010 00000911 
	v_cmp_gt_u32_e32 vcc_lo, s3, v17                           // 000000000BE8: 7C982203 
	scratch_load_b32 v17, off, off offset:580                  // 000000000BEC: DC510244 117C0000 
	s_and_b32 s30, s26, vcc_lo                                 // 000000000BF4: 8B1E6A1A 
	v_lshlrev_b32_e32 v18, 1, v16                              // 000000000BF8: 30242081 
	v_and_b32_e32 v18, 2, v18                                  // 000000000BFC: 36242482 
	s_branch 21                                                // 000000000C00: BFA00015  <_amdgpu_cs_main+0xC58>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000C04: 8C7E077E 
	v_lshlrev_b32_e32 v21, 9, v21                              // 000000000C08: 302A2A89 
	v_lshlrev_b32_e32 v23, 1, v128                             // 000000000C0C: 302F0081 
	v_add_nc_u32_e32 v24, 32, v19                              // 000000000C10: 4A3026A0 
	v_lshlrev_b32_e32 v22, 5, v22                              // 000000000C14: 302C2C85 
	v_cmp_lt_u32_e64 s7, 0xdf, v19                             // 000000000C18: D4490007 000226FF 000000DF 
	v_add_nc_u32_e32 v17, 4, v17                               // 000000000C24: 4A222284 
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000000C28: BF870214 
	v_mov_b32_e32 v19, v24                                     // 000000000C2C: 7E260318 
	v_or3_b32 v21, v21, v22, v23                               // 000000000C30: D6580015 045E2D15 
	s_or_b32 s31, s7, s31                                      // 000000000C38: 8C1F1F07 
	s_waitcnt vmcnt(0)                                         // 000000000C3C: BF8903F7 
	ds_store_b16 v21, v20                                      // 000000000C40: D87C0000 00001415 
	ds_store_b16_d16_hi v21, v20 offset:32                     // 000000000C48: DA840020 00001415 
	s_and_not1_b32 exec_lo, exec_lo, s31                       // 000000000C50: 917E1F7E 
	s_cbranch_execz 57                                         // 000000000C54: BFA50039  <_amdgpu_cs_main+0xD3C>
	v_lshrrev_b32_e32 v21, 7, v19                              // 000000000C58: 322A2687 
	s_waitcnt vmcnt(0)                                         // 000000000C5C: BF8903F7 
	v_and_b32_e32 v22, 14, v17                                 // 000000000C60: 362C228E 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000C64: BF870092 
	v_lshlrev_b32_e32 v20, 4, v21                              // 000000000C68: 30282A84 
	v_or3_b32 v24, v20, s11, v22                               // 000000000C6C: D6580018 04581714 
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000000C74: BF870121 
	v_or_b32_e32 v20, 1, v24                                   // 000000000C78: 38283081 
	v_add_nc_u32_e32 v23, v24, v16                             // 000000000C7C: 4A2E2118 
	v_cmp_gt_u32_e64 s7, s1, v20                               // 000000000C80: D44C0007 00022801 
	s_and_b32 s8, s30, s7                                      // 000000000C88: 8B08071E 
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000000C8C: BF870499 
	s_xor_b32 s8, s8, -1                                       // 000000000C90: 8D08C108 
	s_and_saveexec_b32 s33, s8                                 // 000000000C94: BEA12008 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000C98: BF870009 
	s_xor_b32 s33, exec_lo, s33                                // 000000000C9C: 8D21217E 
	s_cbranch_execz 29                                         // 000000000CA0: BFA5001D  <_amdgpu_cs_main+0xD18>
 <unknown>
	v_cmp_gt_u32_e64 s8, s1, v24                               // 000000000CA8: D44C0008 00023001 
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000CB0: BF8704A2 
 <unknown>
	s_and_b32 s34, vcc_lo, s8                                  // 000000000CB8: 8B22086A 
	s_and_saveexec_b32 s8, s34                                 // 000000000CBC: BE882022 
	s_cbranch_execz 9                                          // 000000000CC0: BFA50009  <_amdgpu_cs_main+0xCE8>
	v_lshlrev_b32_e32 v24, 1, v23                              // 000000000CC4: 30302E81 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000CC8: BF870001 
	v_and_or_b32 v24, v24, -4, v18                             // 000000000CCC: D6570018 04498918 
	s_waitcnt lgkmcnt(0)                                       // 000000000CD4: BF89FC07 
	buffer_load_u16 v24, v24, s[12:15], 0 offen                // 000000000CD8: E0480000 80431818 
	s_waitcnt vmcnt(0)                                         // 000000000CE0: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000CE8: 8C7E087E 
	s_and_b32 s8, vcc_lo, s7                                   // 000000000CEC: 8B08076A 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000CF0: BF870009 
	s_and_saveexec_b32 s7, s8                                  // 000000000CF4: BE872008 
	s_cbranch_execz 6                                          // 000000000CF8: BFA50006  <_amdgpu_cs_main+0xD14>
	v_lshlrev_b32_e32 v23, 1, v23                              // 000000000CFC: 302E2E81 
	s_waitcnt lgkmcnt(0)                                       // 000000000D00: BF89FC07 
	buffer_load_u16 v23, v23, s[12:15], 0 offen offset:2       // 000000000D04: E0480002 80431717 
	s_waitcnt vmcnt(0)                                         // 000000000D0C: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000D14: 8C7E077E 
	s_and_not1_saveexec_b32 s7, s33                            // 000000000D18: BE873021 
	s_cbranch_execz 65465                                      // 000000000D1C: BFA5FFB9  <_amdgpu_cs_main+0xC04>
	v_lshlrev_b32_e32 v20, 1, v23                              // 000000000D20: 30282E81 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000D24: BF870001 
	v_and_b32_e32 v20, -4, v20                                 // 000000000D28: 362828C4 
	s_waitcnt lgkmcnt(0)                                       // 000000000D2C: BF89FC07 
	buffer_load_b32 v20, v20, s[12:15], 0 offen                // 000000000D30: E0500000 80431414 
	s_branch 65458                                             // 000000000D38: BFA0FFB2  <_amdgpu_cs_main+0xC04>
	s_or_b32 exec_lo, exec_lo, s29                             // 000000000D3C: 8C7E1D7E 
	s_mov_b32 s7, 0                                            // 000000000D40: BE870080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000D44: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000000D48: 916A077E 
	s_cbranch_vccnz 95                                         // 000000000D4C: BFA4005F  <_amdgpu_cs_main+0xECC>
	s_and_saveexec_b32 s29, s0                                 // 000000000D50: BE9D2000 
	s_cbranch_execz 92                                         // 000000000D54: BFA5005C  <_amdgpu_cs_main+0xEC8>
	scratch_load_b32 v17, off, off offset:580                  // 000000000D58: DC510244 117C0000 
	v_lshlrev_b32_e32 v16, 1, v129                             // 000000000D60: 30210281 
	v_mov_b32_e32 v18, v129                                    // 000000000D64: 7E240381 
	s_mov_b32 s30, 0                                           // 000000000D68: BE9E0080 
	s_branch 20                                                // 000000000D6C: BFA00014  <_amdgpu_cs_main+0xDC0>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000D70: 8C7E077E 
	v_lshlrev_b32_e32 v20, 5, v20                              // 000000000D74: 30282885 
	v_lshlrev_b32_e32 v21, 1, v21                              // 000000000D78: 302A2A81 
	v_add_nc_u32_e32 v23, 32, v18                              // 000000000D7C: 4A2E24A0 
	v_lshlrev_b32_e32 v19, 9, v19                              // 000000000D80: 30262689 
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v18                         // 000000000D84: 7C9224FF 000000DF 
	v_add_nc_u32_e32 v17, 4, v17                               // 000000000D8C: 4A222284 
	v_add_nc_u32_e32 v16, 64, v16                              // 000000000D90: 4A2020C0 
	v_mov_b32_e32 v18, v23                                     // 000000000D94: 7E240317 
	v_or3_b32 v19, v19, v20, v21                               // 000000000D98: D6580013 04562913 
	s_or_b32 s30, vcc_lo, s30                                  // 000000000DA0: 8C1E1E6A 
	s_waitcnt vmcnt(0)                                         // 000000000DA4: BF8903F7 
	ds_store_b16 v19, v22                                      // 000000000DA8: D87C0000 00001613 
	ds_store_b16_d16_hi v19, v22 offset:2                      // 000000000DB0: DA840002 00001613 
	s_and_not1_b32 exec_lo, exec_lo, s30                       // 000000000DB8: 917E1E7E 
	s_cbranch_execz 66                                         // 000000000DBC: BFA50042  <_amdgpu_cs_main+0xEC8>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000000DC0: BF8701C1 
	v_lshrrev_b32_e32 v19, 7, v18                              // 000000000DC4: 32262487 
	v_and_b32_e32 v21, 14, v16                                 // 000000000DC8: 362A208E 
	s_waitcnt vmcnt(0)                                         // 000000000DCC: BF8903F7 
	v_and_b32_e32 v20, 15, v17                                 // 000000000DD0: 3628228F 
	v_lshlrev_b32_e32 v22, 4, v19                              // 000000000DD4: 302C2684 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000DD8: BF870113 
	v_or_b32_e32 v25, s24, v21                                 // 000000000DDC: 38322A18 
	v_or3_b32 v22, v22, s11, v20                               // 000000000DE0: D6580016 04501716 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000DE8: BF870112 
	v_or_b32_e32 v23, 1, v25                                   // 000000000DEC: 382E3281 
	v_mul_lo_u32 v24, v22, s4                                  // 000000000DF0: D72C0018 00000916 
	v_cmp_gt_u32_e32 vcc_lo, s1, v22                           // 000000000DF8: 7C982C01 
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000DFC: BF8704A3 
	v_cmp_gt_u32_e64 s7, s3, v23                               // 000000000E00: D44C0007 00022E03 
	s_and_b32 s8, s26, vcc_lo                                  // 000000000E08: 8B086A1A 
	s_and_b32 s8, s7, s8                                       // 000000000E0C: 8B080807 
	v_add_nc_u32_e32 v23, v24, v25                             // 000000000E10: 4A2E3318 
	s_xor_b32 s8, s8, -1                                       // 000000000E14: 8D08C108 
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000000E18: BF870499 
	s_and_saveexec_b32 s31, s8                                 // 000000000E1C: BE9F2008 
	s_xor_b32 s31, exec_lo, s31                                // 000000000E20: 8D1F1F7E 
	s_cbranch_execz 31                                         // 000000000E24: BFA5001F  <_amdgpu_cs_main+0xEA4>
 <unknown>
	v_cmp_gt_u32_e64 s8, s3, v25                               // 000000000E2C: D44C0008 00023203 
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000E34: BF8704A2 
 <unknown>
	s_and_b32 s33, s8, vcc_lo                                  // 000000000E3C: 8B216A08 
	s_and_saveexec_b32 s8, s33                                 // 000000000E40: BE882021 
	s_cbranch_execz 11                                         // 000000000E44: BFA5000B  <_amdgpu_cs_main+0xE74>
	v_lshlrev_b32_e32 v24, 1, v24                              // 000000000E48: 30303081 
	v_lshlrev_b32_e32 v25, 1, v23                              // 000000000E4C: 30322E81 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000E50: BF870092 
	v_and_b32_e32 v24, 2, v24                                  // 000000000E54: 36303082 
	v_and_or_b32 v24, v25, -4, v24                             // 000000000E58: D6570018 04618919 
	s_waitcnt lgkmcnt(0)                                       // 000000000E60: BF89FC07 
	buffer_load_u16 v24, v24, s[12:15], 0 offen                // 000000000E64: E0480000 80431818 
	s_waitcnt vmcnt(0)                                         // 000000000E6C: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000E74: 8C7E087E 
	s_and_b32 s8, vcc_lo, s7                                   // 000000000E78: 8B08076A 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000E7C: BF870009 
	s_and_saveexec_b32 s7, s8                                  // 000000000E80: BE872008 
	s_cbranch_execz 6                                          // 000000000E84: BFA50006  <_amdgpu_cs_main+0xEA0>
	v_lshlrev_b32_e32 v23, 1, v23                              // 000000000E88: 302E2E81 
	s_waitcnt lgkmcnt(0)                                       // 000000000E8C: BF89FC07 
	buffer_load_u16 v23, v23, s[12:15], 0 offen offset:2       // 000000000E90: E0480002 80431717 
	s_waitcnt vmcnt(0)                                         // 000000000E98: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000EA0: 8C7E077E 
	s_and_not1_saveexec_b32 s7, s31                            // 000000000EA4: BE87301F 
	s_cbranch_execz 65457                                      // 000000000EA8: BFA5FFB1  <_amdgpu_cs_main+0xD70>
	v_lshlrev_b32_e32 v22, 1, v23                              // 000000000EAC: 302C2E81 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000000EB0: BF870001 
	v_and_b32_e32 v22, -4, v22                                 // 000000000EB4: 362C2CC4 
	s_waitcnt lgkmcnt(0)                                       // 000000000EB8: BF89FC07 
	buffer_load_b32 v22, v22, s[12:15], 0 offen                // 000000000EBC: E0500000 80431616 
	s_branch 65450                                             // 000000000EC4: BFA0FFAA  <_amdgpu_cs_main+0xD70>
	s_or_b32 exec_lo, exec_lo, s29                             // 000000000EC8: 8C7E1D7E 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000ECC: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s9                         // 000000000ED0: 916A097E 
	s_mov_b32 s7, -1                                           // 000000000ED4: BE8700C1 
	s_cbranch_vccnz 93                                         // 000000000ED8: BFA4005D  <_amdgpu_cs_main+0x1050>
	s_and_saveexec_b32 s29, s0                                 // 000000000EDC: BE9D2000 
	s_cbranch_execz 89                                         // 000000000EE0: BFA50059  <_amdgpu_cs_main+0x1048>
	scratch_load_b32 v17, off, off offset:580                  // 000000000EE4: DC510244 117C0000 
	v_or_b32_e32 v16, s10, v128                                // 000000000EEC: 3821000A 
	v_mov_b32_e32 v18, v129                                    // 000000000EF0: 7E240381 
	s_mov_b32 s30, 0                                           // 000000000EF4: BE9E0080 
	s_branch 20                                                // 000000000EF8: BFA00014  <_amdgpu_cs_main+0xF4C>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000000EFC: 8C7E077E 
	v_lshlrev_b32_e32 v20, 9, v20                              // 000000000F00: 30282889 
	v_lshlrev_b32_e32 v22, 1, v128                             // 000000000F04: 302D0081 
	v_add_nc_u32_e32 v23, 32, v18                              // 000000000F08: 4A2E24A0 
	v_lshlrev_b32_e32 v19, 5, v19                              // 000000000F0C: 30262685 
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v18                         // 000000000F10: 7C9224FF 000000DF 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000000F18: BF870193 
	v_dual_mov_b32 v18, v23 :: v_dual_add_nc_u32 v17, 4, v17   // 000000000F1C: CA200117 12102284 
	v_or3_b32 v19, v20, v19, v22                               // 000000000F24: D6580013 045A2714 
	s_or_b32 s30, vcc_lo, s30                                  // 000000000F2C: 8C1E1E6A 
	s_waitcnt vmcnt(0)                                         // 000000000F30: BF8903F7 
	ds_store_b16 v19, v21 offset:1024                          // 000000000F34: D87C0400 00001513 
	ds_store_b16_d16_hi v19, v21 offset:1056                   // 000000000F3C: DA840420 00001513 
	s_and_not1_b32 exec_lo, exec_lo, s30                       // 000000000F44: 917E1E7E 
	s_cbranch_execz 63                                         // 000000000F48: BFA5003F  <_amdgpu_cs_main+0x1048>
	s_waitcnt vmcnt(0)                                         // 000000000F4C: BF8903F7 
	v_and_b32_e32 v19, 14, v17                                 // 000000000F50: 3626228E 
	v_lshrrev_b32_e32 v20, 7, v18                              // 000000000F54: 32282487 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000F58: BF870112 
	v_or_b32_e32 v24, s24, v19                                 // 000000000F5C: 38302618 
	v_lshl_or_b32 v21, v20, 4, v16                             // 000000000F60: D6560015 04410914 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000000F68: BF870112 
	v_or_b32_e32 v22, 1, v24                                   // 000000000F6C: 382C3081 
	v_mul_lo_u32 v23, v21, s5                                  // 000000000F70: D72C0017 00000B15 
	v_cmp_gt_u32_e32 vcc_lo, s2, v21                           // 000000000F78: 7C982A02 
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000000F7C: BF8704C3 
	v_cmp_gt_u32_e64 s7, s3, v22                               // 000000000F80: D44C0007 00022C03 
	s_and_b32 s8, s27, vcc_lo                                  // 000000000F88: 8B086A1B 
	v_add_nc_u32_e32 v22, v23, v24                             // 000000000F8C: 4A2C3117 
	s_and_b32 s8, s7, s8                                       // 000000000F90: 8B080807 
	s_xor_b32 s8, s8, -1                                       // 000000000F94: 8D08C108 
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000000F98: BF870499 
	s_and_saveexec_b32 s31, s8                                 // 000000000F9C: BE9F2008 
	s_xor_b32 s31, exec_lo, s31                                // 000000000FA0: 8D1F1F7E 
	s_cbranch_execz 31                                         // 000000000FA4: BFA5001F  <_amdgpu_cs_main+0x1024>
 <unknown>
	v_cmp_gt_u32_e64 s8, s3, v24                               // 000000000FAC: D44C0008 00023003 
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000000FB4: BF8704A2 
 <unknown>
	s_and_b32 s33, s8, vcc_lo                                  // 000000000FBC: 8B216A08 
	s_and_saveexec_b32 s8, s33                                 // 000000000FC0: BE882021 
	s_cbranch_execz 11                                         // 000000000FC4: BFA5000B  <_amdgpu_cs_main+0xFF4>
	v_lshlrev_b32_e32 v23, 1, v23                              // 000000000FC8: 302E2E81 
	v_lshlrev_b32_e32 v24, 1, v22                              // 000000000FCC: 30302C81 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000000FD0: BF870092 
	v_and_b32_e32 v23, 2, v23                                  // 000000000FD4: 362E2E82 
	v_and_or_b32 v23, v24, -4, v23                             // 000000000FD8: D6570017 045D8918 
	s_waitcnt lgkmcnt(0)                                       // 000000000FE0: BF89FC07 
	buffer_load_u16 v23, v23, s[16:19], 0 offen                // 000000000FE4: E0480000 80441717 
	s_waitcnt vmcnt(0)                                         // 000000000FEC: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s8                              // 000000000FF4: 8C7E087E 
	s_and_b32 s8, vcc_lo, s7                                   // 000000000FF8: 8B08076A 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000000FFC: BF870009 
	s_and_saveexec_b32 s7, s8                                  // 000000001000: BE872008 
	s_cbranch_execz 6                                          // 000000001004: BFA50006  <_amdgpu_cs_main+0x1020>
	v_lshlrev_b32_e32 v22, 1, v22                              // 000000001008: 302C2C81 
	s_waitcnt lgkmcnt(0)                                       // 00000000100C: BF89FC07 
	buffer_load_u16 v22, v22, s[16:19], 0 offen offset:2       // 000000001010: E0480002 80441616 
	s_waitcnt vmcnt(0)                                         // 000000001018: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000001020: 8C7E077E 
	s_and_not1_saveexec_b32 s7, s31                            // 000000001024: BE87301F 
	s_cbranch_execz 65460                                      // 000000001028: BFA5FFB4  <_amdgpu_cs_main+0xEFC>
	v_lshlrev_b32_e32 v21, 1, v22                              // 00000000102C: 302A2C81 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001030: BF870001 
	v_and_b32_e32 v21, -4, v21                                 // 000000001034: 362A2AC4 
	s_waitcnt lgkmcnt(0)                                       // 000000001038: BF89FC07 
	buffer_load_b32 v21, v21, s[16:19], 0 offen                // 00000000103C: E0500000 80441515 
	s_branch 65453                                             // 000000001044: BFA0FFAD  <_amdgpu_cs_main+0xEFC>
	s_or_b32 exec_lo, exec_lo, s29                             // 000000001048: 8C7E1D7E 
	s_mov_b32 s7, 0                                            // 00000000104C: BE870080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001050: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000001054: 916A077E 
	s_cbranch_vccnz 95                                         // 000000001058: BFA4005F  <_amdgpu_cs_main+0x11D8>
	s_and_saveexec_b32 s8, s0                                  // 00000000105C: BE882000 
	s_cbranch_execz 92                                         // 000000001060: BFA5005C  <_amdgpu_cs_main+0x11D4>
	scratch_load_b32 v17, off, off offset:580                  // 000000001064: DC510244 117C0000 
	v_lshlrev_b32_e32 v16, 1, v129                             // 00000000106C: 30210281 
	v_mov_b32_e32 v18, v129                                    // 000000001070: 7E240381 
	s_mov_b32 s29, 0                                           // 000000001074: BE9D0080 
	s_branch 20                                                // 000000001078: BFA00014  <_amdgpu_cs_main+0x10CC>
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000107C: 8C7E007E 
	v_lshlrev_b32_e32 v21, 5, v21                              // 000000001080: 302A2A85 
	v_lshlrev_b32_e32 v20, 1, v20                              // 000000001084: 30282881 
	v_add_nc_u32_e32 v23, 32, v18                              // 000000001088: 4A2E24A0 
	v_lshlrev_b32_e32 v19, 9, v19                              // 00000000108C: 30262689 
	v_cmp_lt_u32_e32 vcc_lo, 0xdf, v18                         // 000000001090: 7C9224FF 000000DF 
	v_add_nc_u32_e32 v17, 4, v17                               // 000000001098: 4A222284 
	v_add_nc_u32_e32 v16, 64, v16                              // 00000000109C: 4A2020C0 
	v_mov_b32_e32 v18, v23                                     // 0000000010A0: 7E240317 
	v_or3_b32 v19, v19, v21, v20                               // 0000000010A4: D6580013 04522B13 
	s_or_b32 s29, vcc_lo, s29                                  // 0000000010AC: 8C1D1D6A 
	s_waitcnt vmcnt(0)                                         // 0000000010B0: BF8903F7 
	ds_store_b16 v19, v22 offset:1024                          // 0000000010B4: D87C0400 00001613 
	ds_store_b16_d16_hi v19, v22 offset:1026                   // 0000000010BC: DA840402 00001613 
	s_and_not1_b32 exec_lo, exec_lo, s29                       // 0000000010C4: 917E1D7E 
	s_cbranch_execz 66                                         // 0000000010C8: BFA50042  <_amdgpu_cs_main+0x11D4>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000010CC: BF8701C1 
	v_lshrrev_b32_e32 v19, 7, v18                              // 0000000010D0: 32262487 
	v_and_b32_e32 v20, 14, v16                                 // 0000000010D4: 3628208E 
	s_waitcnt vmcnt(0)                                         // 0000000010D8: BF8903F7 
	v_and_b32_e32 v21, 15, v17                                 // 0000000010DC: 362A228F 
	v_lshlrev_b32_e32 v22, 4, v19                              // 0000000010E0: 302C2684 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000010E4: BF870191 
	v_or3_b32 v25, v22, s10, v20                               // 0000000010E8: D6580019 04501516 
	v_or_b32_e32 v22, s24, v21                                 // 0000000010F0: 382C2A18 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000010F4: BF870112 
	v_or_b32_e32 v23, 1, v25                                   // 0000000010F8: 382E3281 
	v_mul_lo_u32 v24, v22, s5                                  // 0000000010FC: D72C0018 00000B16 
	v_cmp_gt_u32_e32 vcc_lo, s3, v22                           // 000000001104: 7C982C03 
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001108: BF8704C3 
	v_cmp_gt_u32_e64 s0, s2, v23                               // 00000000110C: D44C0000 00022E02 
	s_and_b32 s7, s27, vcc_lo                                  // 000000001114: 8B076A1B 
	v_add_nc_u32_e32 v23, v24, v25                             // 000000001118: 4A2E3318 
	s_and_b32 s7, s7, s0                                       // 00000000111C: 8B070007 
	s_xor_b32 s7, s7, -1                                       // 000000001120: 8D07C107 
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001124: BF870499 
	s_and_saveexec_b32 s30, s7                                 // 000000001128: BE9E2007 
	s_xor_b32 s30, exec_lo, s30                                // 00000000112C: 8D1E1E7E 
	s_cbranch_execz 31                                         // 000000001130: BFA5001F  <_amdgpu_cs_main+0x11B0>
 <unknown>
	v_cmp_gt_u32_e64 s7, s2, v25                               // 000000001138: D44C0007 00023202 
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001140: BF8704A2 
 <unknown>
	s_and_b32 s31, vcc_lo, s7                                  // 000000001148: 8B1F076A 
	s_and_saveexec_b32 s7, s31                                 // 00000000114C: BE87201F 
	s_cbranch_execz 11                                         // 000000001150: BFA5000B  <_amdgpu_cs_main+0x1180>
	v_lshlrev_b32_e32 v24, 1, v24                              // 000000001154: 30303081 
	v_lshlrev_b32_e32 v25, 1, v23                              // 000000001158: 30322E81 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000115C: BF870092 
	v_and_b32_e32 v24, 2, v24                                  // 000000001160: 36303082 
	v_and_or_b32 v24, v25, -4, v24                             // 000000001164: D6570018 04618919 
	s_waitcnt lgkmcnt(0)                                       // 00000000116C: BF89FC07 
	buffer_load_u16 v24, v24, s[16:19], 0 offen                // 000000001170: E0480000 80441818 
	s_waitcnt vmcnt(0)                                         // 000000001178: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000001180: 8C7E077E 
	s_and_b32 s7, vcc_lo, s0                                   // 000000001184: 8B07006A 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001188: BF870009 
	s_and_saveexec_b32 s0, s7                                  // 00000000118C: BE802007 
	s_cbranch_execz 6                                          // 000000001190: BFA50006  <_amdgpu_cs_main+0x11AC>
	v_lshlrev_b32_e32 v23, 1, v23                              // 000000001194: 302E2E81 
	s_waitcnt lgkmcnt(0)                                       // 000000001198: BF89FC07 
	buffer_load_u16 v23, v23, s[16:19], 0 offen offset:2       // 00000000119C: E0480002 80441717 
	s_waitcnt vmcnt(0)                                         // 0000000011A4: BF8903F7 
 <unknown>
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000011AC: 8C7E007E 
	s_and_not1_saveexec_b32 s0, s30                            // 0000000011B0: BE80301E 
	s_cbranch_execz 65457                                      // 0000000011B4: BFA5FFB1  <_amdgpu_cs_main+0x107C>
	v_lshlrev_b32_e32 v22, 1, v23                              // 0000000011B8: 302C2E81 
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011BC: BF870001 
	v_and_b32_e32 v22, -4, v22                                 // 0000000011C0: 362C2CC4 
	s_waitcnt lgkmcnt(0)                                       // 0000000011C4: BF89FC07 
	buffer_load_b32 v22, v22, s[16:19], 0 offen                // 0000000011C8: E0500000 80441616 
	s_branch 65450                                             // 0000000011D0: BFA0FFAA  <_amdgpu_cs_main+0x107C>
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000011D4: 8C7E087E 
	v_ashrrev_i32_e32 v16, 31, v129                            // 0000000011D8: 3421029F 
	s_waitcnt lgkmcnt(0)                                       // 0000000011DC: BF89FC07 
	s_waitcnt_vscnt null, 0x0                                  // 0000000011E0: BC7C0000 
	s_and_b32 s0, s28, 7                                       // 0000000011E4: 8B00871C 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000011E8: BF870091 
	v_lshrrev_b32_e32 v16, 28, v16                             // 0000000011EC: 3220209C 
	v_add_nc_u32_e32 v16, v129, v16                            // 0000000011F0: 4A202181 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000011F4: BF870091 
	v_and_b32_e32 v16, -16, v16                                // 0000000011F8: 362020D0 
	v_sub_nc_u32_e32 v16, v129, v16                            // 0000000011FC: 4C202181 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001200: BF870001 
	v_lshlrev_b32_e32 v17, 1, v16                              // 000000001204: 30222081 
	v_lshlrev_b32_e32 v16, 5, v16                              // 000000001208: 30202085 
	ds_load_u16_d16 v40, v17 offset:1024                       // 00000000120C: DA980400 28000011 
	ds_load_u16_d16 v41, v17 offset:1088                       // 000000001214: DA980440 29000011 
	ds_load_u16_d16 v42, v17 offset:1152                       // 00000000121C: DA980480 2A000011 
	ds_load_u16_d16 v43, v17 offset:1216                       // 000000001224: DA9804C0 2B000011 
	ds_load_u16_d16 v44, v17 offset:1280                       // 00000000122C: DA980500 2C000011 
	ds_load_u16_d16 v45, v17 offset:1344                       // 000000001234: DA980540 2D000011 
	ds_load_u16_d16 v46, v17 offset:1408                       // 00000000123C: DA980580 2E000011 
	ds_load_u16_d16 v47, v17 offset:1472                       // 000000001244: DA9805C0 2F000011 
	ds_load_u16_d16 v0, v17 offset:1536                        // 00000000124C: DA980600 00000011 
	ds_load_u16_d16 v1, v17 offset:1600                        // 000000001254: DA980640 01000011 
	ds_load_u16_d16 v2, v17 offset:1664                        // 00000000125C: DA980680 02000011 
	ds_load_u16_d16 v3, v17 offset:1728                        // 000000001264: DA9806C0 03000011 
	ds_load_u16_d16 v4, v17 offset:1792                        // 00000000126C: DA980700 04000011 
	ds_load_u16_d16 v5, v17 offset:1856                        // 000000001274: DA980740 05000011 
	ds_load_u16_d16 v6, v17 offset:1920                        // 00000000127C: DA980780 06000011 
	ds_load_u16_d16 v7, v17 offset:1984                        // 000000001284: DA9807C0 07000011 
	ds_load_b128 v[18:21], v16                                 // 00000000128C: DBFC0000 12000010 
	ds_load_b128 v[22:25], v16 offset:16                       // 000000001294: DBFC0010 16000010 
	s_waitcnt lgkmcnt(17)                                      // 00000000129C: BF89FD17 
	ds_load_u16_d16_hi v40, v17 offset:1056                    // 0000000012A0: DA9C0420 28000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012A8: BF89FD17 
	ds_load_u16_d16_hi v41, v17 offset:1120                    // 0000000012AC: DA9C0460 29000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012B4: BF89FD17 
	ds_load_u16_d16_hi v42, v17 offset:1184                    // 0000000012B8: DA9C04A0 2A000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012C0: BF89FD17 
	ds_load_u16_d16_hi v43, v17 offset:1248                    // 0000000012C4: DA9C04E0 2B000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012CC: BF89FD17 
	ds_load_u16_d16_hi v44, v17 offset:1312                    // 0000000012D0: DA9C0520 2C000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012D8: BF89FD17 
	ds_load_u16_d16_hi v45, v17 offset:1376                    // 0000000012DC: DA9C0560 2D000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012E4: BF89FD17 
	ds_load_u16_d16_hi v46, v17 offset:1440                    // 0000000012E8: DA9C05A0 2E000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012F0: BF89FD17 
	ds_load_u16_d16_hi v47, v17 offset:1504                    // 0000000012F4: DA9C05E0 2F000011 
	s_waitcnt lgkmcnt(17)                                      // 0000000012FC: BF89FD17 
	ds_load_u16_d16_hi v0, v17 offset:1568                     // 000000001300: DA9C0620 00000011 
	s_waitcnt lgkmcnt(17)                                      // 000000001308: BF89FD17 
	ds_load_u16_d16_hi v1, v17 offset:1632                     // 00000000130C: DA9C0660 01000011 
	s_waitcnt lgkmcnt(17)                                      // 000000001314: BF89FD17 
	ds_load_u16_d16_hi v2, v17 offset:1696                     // 000000001318: DA9C06A0 02000011 
	s_waitcnt lgkmcnt(17)                                      // 000000001320: BF89FD17 
	ds_load_u16_d16_hi v3, v17 offset:1760                     // 000000001324: DA9C06E0 03000011 
	s_waitcnt lgkmcnt(17)                                      // 00000000132C: BF89FD17 
	ds_load_u16_d16_hi v4, v17 offset:1824                     // 000000001330: DA9C0720 04000011 
	s_waitcnt lgkmcnt(17)                                      // 000000001338: BF89FD17 
	ds_load_u16_d16_hi v5, v17 offset:1888                     // 00000000133C: DA9C0760 05000011 
	s_waitcnt lgkmcnt(17)                                      // 000000001344: BF89FD17 
	ds_load_u16_d16_hi v6, v17 offset:1952                     // 000000001348: DA9C07A0 06000011 
	s_waitcnt lgkmcnt(17)                                      // 000000001350: BF89FD17 
	ds_load_u16_d16_hi v7, v17 offset:2016                     // 000000001354: DA9C07E0 07000011 
	s_waitcnt lgkmcnt(17)                                      // 00000000135C: BF89FD17 
	scratch_store_b128 off, v[18:21], off offset:352           // 000000001360: DC750160 007C1200 
	s_waitcnt lgkmcnt(16)                                      // 000000001368: BF89FD07 
	scratch_store_b128 off, v[22:25], off offset:368           // 00000000136C: DC750170 007C1600 
	ds_load_b128 v[18:21], v16 offset:512                      // 000000001374: DBFC0200 12000010 
	ds_load_b128 v[22:25], v16 offset:528                      // 00000000137C: DBFC0210 16000010 
	s_waitcnt lgkmcnt(14)                                      // 000000001384: BF89FCE7 
	scratch_store_b128 off, v[40:43], off offset:420           // 000000001388: DC7501A4 007C2800 
	s_waitcnt lgkmcnt(10)                                      // 000000001390: BF89FCA7 
	scratch_store_b128 off, v[44:47], off offset:436           // 000000001394: DC7501B4 007C2C00 
	s_waitcnt lgkmcnt(6)                                       // 00000000139C: BF89FC67 
	scratch_store_b128 off, v[0:3], off offset:388             // 0000000013A0: DC750184 007C0000 
	s_waitcnt lgkmcnt(2)                                       // 0000000013A8: BF89FC27 
	scratch_store_b128 off, v[4:7], off offset:404             // 0000000013AC: DC750194 007C0400 
	s_waitcnt lgkmcnt(1)                                       // 0000000013B4: BF89FC17 
	scratch_store_b128 off, v[18:21], off offset:320           // 0000000013B8: DC750140 007C1200 
	s_waitcnt lgkmcnt(0)                                       // 0000000013C0: BF89FC07 
	scratch_store_b128 off, v[22:25], off offset:336           // 0000000013C4: DC750150 007C1600 
	s_cbranch_scc0 3949                                        // 0000000013CC: BFA10F6D  <_amdgpu_cs_main+0x5184>
	s_cmp_lg_u32 s0, 1                                         // 0000000013D0: BF078100 
	s_mov_b32 s7, -1                                           // 0000000013D4: BE8700C1 
	s_cbranch_scc0 3445                                        // 0000000013D8: BFA10D75  <_amdgpu_cs_main+0x49B0>
	s_cmp_lg_u32 s0, 2                                         // 0000000013DC: BF078200 
	s_cbranch_scc0 2901                                        // 0000000013E0: BFA10B55  <_amdgpu_cs_main+0x4138>
	s_cmp_lg_u32 s0, 3                                         // 0000000013E4: BF078300 
	s_cbranch_scc0 2330                                        // 0000000013E8: BFA1091A  <_amdgpu_cs_main+0x3854>
	s_cmp_lg_u32 s0, 4                                         // 0000000013EC: BF078400 
	s_cbranch_scc0 1822                                        // 0000000013F0: BFA1071E  <_amdgpu_cs_main+0x306C>
	s_cmp_lg_u32 s0, 5                                         // 0000000013F4: BF078500 
	s_cbranch_scc0 1296                                        // 0000000013F8: BFA10510  <_amdgpu_cs_main+0x283C>
	s_cmp_lg_u32 s0, 6                                         // 0000000013FC: BF078600 
	s_mov_b32 s0, -1                                           // 000000001400: BE8000C1 
	s_cbranch_scc0 637                                         // 000000001404: BFA1027D  <_amdgpu_cs_main+0x1DFC>
	s_clause 0x7                                               // 000000001408: BF850007 
	scratch_load_b128 v[97:100], off, off offset:192           // 00000000140C: DC5D00C0 617C0000 
	scratch_load_b128 v[101:104], off, off offset:208          // 000000001414: DC5D00D0 657C0000 
	scratch_load_b128 v[168:171], off, off offset:352          // 00000000141C: DC5D0160 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:368          // 000000001424: DC5D0170 AC7C0000 
	scratch_load_b128 v[0:3], off, off offset:420              // 00000000142C: DC5D01A4 007C0000 
	scratch_load_b128 v[4:7], off, off offset:436              // 000000001434: DC5D01B4 047C0000 
	scratch_load_b128 v[24:27], off, off offset:320            // 00000000143C: DC5D0140 187C0000 
	scratch_load_b128 v[28:31], off, off offset:336            // 000000001444: DC5D0150 1C7C0000 
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 00000000144C: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 000000001454: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 00000000145C: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 000000001464: CA100121 A1A00120 
	s_clause 0x1                                               // 00000000146C: BF850001 
	scratch_store_b128 off, v[184:187], off offset:584         // 000000001470: DC750248 007CB800 
	scratch_store_b128 off, v[188:191], off offset:600         // 000000001478: DC750258 007CBC00 
	s_clause 0x5                                               // 000000001480: BF850005 
	scratch_load_b128 v[145:148], off, off offset:160          // 000000001484: DC5D00A0 917C0000 
	scratch_load_b128 v[149:152], off, off offset:176          // 00000000148C: DC5D00B0 957C0000 
	scratch_load_b128 v[200:203], off, off offset:96           // 000000001494: DC5D0060 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:112          // 00000000149C: DC5D0070 CC7C0000 
	scratch_load_b128 v[176:179], off, off offset:288          // 0000000014A4: DC5D0120 B07C0000 
	scratch_load_b128 v[180:183], off, off offset:304          // 0000000014AC: DC5D0130 B47C0000 
	v_dual_mov_b32 v192, v232 :: v_dual_mov_b32 v193, v233     // 0000000014B4: CA1001E8 C0C001E9 
	v_dual_mov_b32 v194, v234 :: v_dual_mov_b32 v195, v235     // 0000000014BC: CA1001EA C2C201EB 
	v_dual_mov_b32 v196, v236 :: v_dual_mov_b32 v197, v237     // 0000000014C4: CA1001EC C4C401ED 
	v_dual_mov_b32 v198, v238 :: v_dual_mov_b32 v199, v239     // 0000000014CC: CA1001EE C6C601EF 
	s_clause 0x7                                               // 0000000014D4: BF850007 
	scratch_load_b128 v[232:235], off, off offset:256          // 0000000014D8: DC5D0100 E87C0000 
	scratch_load_b128 v[236:239], off, off offset:272          // 0000000014E0: DC5D0110 EC7C0000 
	scratch_load_b128 v[184:187], off, off offset:584          // 0000000014E8: DC5D0248 B87C0000 
	scratch_load_b128 v[188:191], off, off offset:600          // 0000000014F0: DC5D0258 BC7C0000 
	scratch_load_b128 v[128:131], off, off offset:32           // 0000000014F8: DC5D0020 807C0000 
	scratch_load_b128 v[132:135], off, off offset:48           // 000000001500: DC5D0030 847C0000 
	scratch_load_b128 v[72:75], off, off offset:64             // 000000001508: DC5D0040 487C0000 
	scratch_load_b128 v[76:79], off, off offset:80             // 000000001510: DC5D0050 4C7C0000 
	s_mov_b32 s0, 0                                            // 000000001518: BE800080 
	v_dual_mov_b32 v65, v199 :: v_dual_mov_b32 v64, v198       // 00000000151C: CA1001C7 414001C6 
	v_dual_mov_b32 v63, v197 :: v_dual_mov_b32 v62, v196       // 000000001524: CA1001C5 3F3E01C4 
	v_dual_mov_b32 v61, v195 :: v_dual_mov_b32 v60, v194       // 00000000152C: CA1001C3 3D3C01C2 
	v_dual_mov_b32 v59, v193 :: v_dual_mov_b32 v58, v192       // 000000001534: CA1001C1 3B3A01C0 
	v_dual_mov_b32 v112, v223 :: v_dual_mov_b32 v111, v222     // 00000000153C: CA1001DF 706E01DE 
	v_dual_mov_b32 v110, v221 :: v_dual_mov_b32 v109, v220     // 000000001544: CA1001DD 6E6C01DC 
	v_dual_mov_b32 v108, v219 :: v_dual_mov_b32 v107, v218     // 00000000154C: CA1001DB 6C6A01DA 
	v_dual_mov_b32 v106, v217 :: v_dual_mov_b32 v105, v216     // 000000001554: CA1001D9 6A6801D8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(16)                                        // 000000001584: BF8943F7 
	v_wmma_f16_16x16x16_f16 v[97:104], v[168:175], v[0:7], v[97:104]// 000000001588: CC424061 1D8601A8 
	s_waitcnt vmcnt(14)                                        // 000000001590: BF893BF7 
	v_wmma_f16_16x16x16_f16 v[160:167], v[24:31], v[0:7], v[160:167]// 000000001594: CC4240A0 1E820118 
	v_dual_mov_b32 v0, v137 :: v_dual_mov_b32 v1, v138         // 00000000159C: CA100189 0000018A 
	v_dual_mov_b32 v2, v139 :: v_dual_mov_b32 v3, v140         // 0000000015A4: CA10018B 0202018C 
	v_dual_mov_b32 v4, v141 :: v_dual_mov_b32 v5, v142         // 0000000015AC: CA10018D 0404018E 
	v_dual_mov_b32 v6, v143 :: v_dual_mov_b32 v7, v144         // 0000000015B4: CA10018F 06060190 
	s_clause 0x1                                               // 0000000015BC: BF850001 
	scratch_load_b128 v[137:140], off, off offset:388          // 0000000015C0: DC5D0184 897C0000 
	scratch_load_b128 v[141:144], off, off offset:404          // 0000000015C8: DC5D0194 8D7C0000 
	s_waitcnt vmcnt(14)                                        // 0000000015D0: BF893BF7 
	v_dual_mov_b32 v96, v152 :: v_dual_mov_b32 v95, v151       // 0000000015D4: CA100198 605E0197 
	v_dual_mov_b32 v94, v150 :: v_dual_mov_b32 v93, v149       // 0000000015DC: CA100196 5E5C0195 
	v_dual_mov_b32 v92, v148 :: v_dual_mov_b32 v91, v147       // 0000000015E4: CA100194 5C5A0193 
	v_dual_mov_b32 v90, v146 :: v_dual_mov_b32 v89, v145       // 0000000015EC: CA100192 5A580191 
	s_waitcnt vmcnt(12)                                        // 0000000015F4: BF8933F7 
	v_dual_mov_b32 v57, v207 :: v_dual_mov_b32 v56, v206       // 0000000015F8: CA1001CF 393801CE 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(4)                                         // 000000001618: BF8913F7 
	v_dual_mov_b32 v23, v135 :: v_dual_mov_b32 v22, v134       // 00000000161C: CA100187 17160186 
	v_dual_mov_b32 v21, v133 :: v_dual_mov_b32 v20, v132       // 000000001624: CA100185 15140184 
	v_dual_mov_b32 v88, v239 :: v_dual_mov_b32 v87, v238       // 00000000162C: CA1001EF 585601EE 
	v_dual_mov_b32 v86, v237 :: v_dual_mov_b32 v85, v236       // 000000001634: CA1001ED 565401EC 
	v_dual_mov_b32 v84, v235 :: v_dual_mov_b32 v83, v234       // 00000000163C: CA1001EB 545201EA 
	v_dual_mov_b32 v82, v233 :: v_dual_mov_b32 v81, v232       // 000000001644: CA1001E9 525001E8 
	v_dual_mov_b32 v53, v203 :: v_dual_mov_b32 v52, v202       // 00000000164C: CA1001CB 353401CA 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v19, v131 :: v_dual_mov_b32 v18, v130       // 00000000167C: CA100183 13120182 
	s_waitcnt vmcnt(3)                                         // 000000001684: BF890FF7 
	v_dual_mov_b32 v209, v73 :: v_dual_mov_b32 v208, v72       // 000000001688: CA100149 D1D00148 
	v_dual_mov_b32 v17, v129 :: v_dual_mov_b32 v16, v128       // 000000001690: CA100181 11100180 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(2)                                         // 0000000016A0: BF890BF7 
	v_dual_mov_b32 v215, v79 :: v_dual_mov_b32 v214, v78       // 0000000016A4: CA10014F D7D6014E 
	v_dual_mov_b32 v213, v77 :: v_dual_mov_b32 v212, v76       // 0000000016AC: CA10014D D5D4014C 
	v_dual_mov_b32 v211, v75 :: v_dual_mov_b32 v210, v74       // 0000000016B4: CA10014B D3D2014A 
	v_dual_mov_b32 v55, v205 :: v_dual_mov_b32 v54, v204       // 0000000016BC: CA1001CD 373601CC 
	v_dual_mov_b32 v51, v201 :: v_dual_mov_b32 v50, v200       // 0000000016C4: CA1001C9 333201C8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v128, off offset:384                // 000000001748: DC690180 007C8000 
	s_waitcnt vmcnt(0)                                         // 000000001750: BF8903F7 
	v_wmma_f16_16x16x16_f16 v[97:104], v[168:175], v[137:144], v[97:104] op_sel:[0,0,1]// 000000001754: CC426061 1D8713A8 
	s_clause 0x3                                               // 00000000175C: BF850003 
	scratch_load_b128 v[113:116], off, off offset:584          // 000000001760: DC5D0248 717C0000 
	scratch_load_b128 v[117:120], off, off offset:600          // 000000001768: DC5D0258 757C0000 
	scratch_load_b128 v[168:171], off, off offset:584          // 000000001770: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001778: DC5D0258 AC7C0000 
	v_wmma_f16_16x16x16_f16 v[160:167], v[24:31], v[137:144], v[160:167] op_sel:[0,0,1]// 000000001780: CC4260A0 1E831318 
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6           // 000000001788: CA100107 1F1E0106 
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4           // 000000001790: CA100105 1D1C0104 
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2           // 000000001798: CA100103 1B1A0102 
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v24, v0           // 0000000017A0: CA100101 19180100 
	s_clause 0x3                                               // 0000000017A8: BF850003 
	scratch_load_b128 v[0:3], off, off offset:128              // 0000000017AC: DC5D0080 007C0000 
	scratch_load_b128 v[4:7], off, off offset:144              // 0000000017B4: DC5D0090 047C0000 
	scratch_load_b128 v[137:140], off, off offset:224          // 0000000017BC: DC5D00E0 897C0000 
	scratch_load_b128 v[141:144], off, off offset:240          // 0000000017C4: DC5D00F0 8D7C0000 
	v_dual_mov_b32 v49, v31 :: v_dual_mov_b32 v48, v30         // 0000000017CC: CA10011F 3130011E 
	v_dual_mov_b32 v43, v25 :: v_dual_mov_b32 v42, v24         // 0000000017D4: CA100119 2B2A0118 
	v_dual_mov_b32 v47, v29 :: v_dual_mov_b32 v46, v28         // 0000000017DC: CA10011D 2F2E011C 
	v_dual_mov_b32 v45, v27 :: v_dual_mov_b32 v44, v26         // 0000000017E4: CA10011B 2D2C011A 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(4)                                         // 000000001810: BF8913F7 
	v_mov_b32_e32 v119, v174                                   // 000000001814: 7EEE03AE 
	s_clause 0x1                                               // 000000001818: BF850001 
	scratch_load_b128 v[168:171], off, off offset:584          // 00000000181C: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001824: DC5D0258 AC7C0000 
	s_waitcnt vmcnt(5)                                         // 00000000182C: BF8917F7 
	v_dual_mov_b32 v249, v1 :: v_dual_mov_b32 v248, v0         // 000000001830: CA100101 F9F80100 
 <unknown>
	s_waitcnt vmcnt(2)                                         // 00000000183C: BF890BF7 
	v_dual_mov_b32 v247, v144 :: v_dual_mov_b32 v246, v143     // 000000001840: CA100190 F7F6018F 
	v_dual_mov_b32 v245, v142 :: v_dual_mov_b32 v244, v141     // 000000001848: CA10018E F5F4018D 
	scratch_store_b32 off, v0, off offset:648                  // 000000001850: DC690288 007C0000 
 <unknown>
	v_dual_mov_b32 v243, v140 :: v_dual_mov_b32 v242, v139     // 00000000185C: CA10018C F3F2018B 
	v_dual_mov_b32 v241, v138 :: v_dual_mov_b32 v240, v137     // 000000001864: CA10018A F1F00189 
	scratch_store_b32 off, v0, off offset:552                  // 00000000186C: DC690228 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v255, v7 :: v_dual_mov_b32 v254, v6         // 000000001890: CA100107 FFFE0106 
	scratch_store_b32 off, v0, off offset:644                  // 000000001898: DC690284 007C0000 
 <unknown>
	v_dual_mov_b32 v253, v5 :: v_dual_mov_b32 v252, v4         // 0000000018A4: CA100105 FDFC0104 
	v_dual_mov_b32 v251, v3 :: v_dual_mov_b32 v250, v2         // 0000000018AC: CA100103 FBFA0102 
	scratch_store_b32 off, v0, off offset:680                  // 0000000018B4: DC6902A8 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:716                  // 0000000018D8: DC6902CC 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:712                  // 000000001908: DC6902C8 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:624                  // 000000001928: DC690270 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:620                  // 000000001938: DC69026C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:708                  // 000000001948: DC6902C4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:704                  // 000000001958: DC6902C0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:700                  // 000000001968: DC6902BC 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:696                  // 000000001978: DC6902B8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:536                  // 000000001988: DC690218 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 000000001998: DC690204 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:652                  // 0000000019A8: DC69028C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:656                  // 0000000019B8: DC690290 007C0000 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 0000000019C8: BF850001 
	scratch_load_b128 v[145:148], off, off                     // 0000000019CC: DC5D0000 917C0000 
	scratch_load_b128 v[149:152], off, off offset:16           // 0000000019D4: DC5D0010 957C0000 
	s_waitcnt vmcnt(0)                                         // 0000000019DC: BF8903F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:660                  // 0000000019F8: DC690294 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:576                  // 000000001A08: DC690240 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:568                  // 000000001A18: DC690238 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:496                  // 000000001A28: DC6901F0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:480                  // 000000001A38: DC6901E0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:748                  // 000000001A48: DC6902EC 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:744                  // 000000001A58: DC6902E8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:740                  // 000000001A68: DC6902E4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:736                  // 000000001A78: DC6902E0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:732                  // 000000001A88: DC6902DC 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:728                  // 000000001A98: DC6902D8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:724                  // 000000001AA8: DC6902D4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:720                  // 000000001AB8: DC6902D0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:640                  // 000000001AC8: DC690280 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:636                  // 000000001AD8: DC69027C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:548                  // 000000001AE8: DC690224 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:476                  // 000000001AF8: DC6901DC 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v144, v31 :: v_dual_mov_b32 v143, v30       // 000000001B08: CA10011F 908E011E 
	v_dual_mov_b32 v142, v29 :: v_dual_mov_b32 v141, v28       // 000000001B10: CA10011D 8E8C011C 
	v_dual_mov_b32 v140, v27 :: v_dual_mov_b32 v139, v26       // 000000001B18: CA10011B 8C8A011A 
	v_dual_mov_b32 v138, v25 :: v_dual_mov_b32 v137, v24       // 000000001B20: CA100119 8A880118 
 <unknown>
 <unknown>
	v_mov_b32_e32 v118, v173                                   // 000000001B30: 7EEC03AD 
	s_clause 0x1                                               // 000000001B34: BF850001 
	scratch_load_b128 v[168:171], off, off offset:584          // 000000001B38: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001B40: DC5D0258 AC7C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:472                  // 000000001B50: DC6901D8 007C0000 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v25, off offset:540                 // 000000001B64: DC69021C 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:468                  // 000000001B74: DC6901D4 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:528                 // 000000001B80: DC690210 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:464                  // 000000001B90: DC6901D0 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:524                 // 000000001B9C: DC69020C 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v25, off offset:544                 // 000000001BAC: DC690220 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:460                  // 000000001BBC: DC6901CC 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:512                 // 000000001BC8: DC690200 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:456                  // 000000001BD8: DC6901C8 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:508                 // 000000001BE4: DC6901FC 007C1900 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000001BF4: BF850001 
	scratch_store_b32 off, v0, off offset:452                  // 000000001BF8: DC6901C4 007C0000 
	scratch_store_b32 off, v25, off offset:504                 // 000000001C00: DC6901F8 007C1900 
	s_waitcnt vmcnt(0)                                         // 000000001C08: BF8903F7 
	v_mov_b32_e32 v117, v172                                   // 000000001C0C: 7EEA03AC 
	s_clause 0x1                                               // 000000001C10: BF850001 
	scratch_load_b128 v[168:171], off, off offset:584          // 000000001C14: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001C1C: DC5D0258 AC7C0000 
	s_waitcnt vmcnt(1)                                         // 000000001C24: BF8907F7 
	v_mov_b32_e32 v116, v171                                   // 000000001C28: 7EE803AB 
	s_clause 0x1                                               // 000000001C2C: BF850001 
	scratch_load_b128 v[168:171], off, off offset:584          // 000000001C30: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001C38: DC5D0258 AC7C0000 
	s_waitcnt vmcnt(1)                                         // 000000001C40: BF8907F7 
	v_mov_b32_e32 v115, v170                                   // 000000001C44: 7EE603AA 
	s_clause 0x1                                               // 000000001C48: BF850001 
	scratch_load_b128 v[168:171], off, off offset:584          // 000000001C4C: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001C54: DC5D0258 AC7C0000 
	s_waitcnt vmcnt(1)                                         // 000000001C5C: BF8907F7 
	v_mov_b32_e32 v114, v169                                   // 000000001C60: 7EE403A9 
	s_clause 0x1                                               // 000000001C64: BF850001 
	scratch_load_b128 v[168:171], off, off offset:584          // 000000001C68: DC5D0248 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:600          // 000000001C70: DC5D0258 AC7C0000 
	s_waitcnt vmcnt(1)                                         // 000000001C78: BF8907F7 
	v_mov_b32_e32 v171, v179                                   // 000000001C7C: 7F5603B3 
	v_mov_b32_e32 v169, v177                                   // 000000001C80: 7F5203B1 
	s_waitcnt vmcnt(0)                                         // 000000001C84: BF8903F7 
	v_dual_mov_b32 v175, v183 :: v_dual_mov_b32 v172, v180     // 000000001C88: CA1001B7 AFAC01B4 
	v_dual_mov_b32 v174, v182 :: v_dual_mov_b32 v173, v181     // 000000001C90: CA1001B6 AEAC01B5 
	v_mov_b32_e32 v170, v178                                   // 000000001C98: 7F5403B2 
 <unknown>
 <unknown>
	v_mov_b32_e32 v113, v168                                   // 000000001CA4: 7EE203A8 
	v_mov_b32_e32 v168, v176                                   // 000000001CA8: 7F5003B0 
 <unknown>
 <unknown>
	v_dual_mov_b32 v239, v199 :: v_dual_mov_b32 v236, v196     // 000000001CB4: CA1001C7 EFEC01C4 
	v_mov_b32_e32 v233, v193                                   // 000000001CBC: 7FD203C1 
	v_dual_mov_b32 v235, v195 :: v_dual_mov_b32 v232, v192     // 000000001CC0: CA1001C3 EBE801C0 
	v_dual_mov_b32 v238, v198 :: v_dual_mov_b32 v237, v197     // 000000001CC8: CA1001C6 EEEC01C5 
	v_mov_b32_e32 v234, v194                                   // 000000001CD0: 7FD403C2 
 <unknown>
 <unknown>
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001CDC: BF870214 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:560                  // 000000001D08: DC690230 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:564                  // 000000001D18: DC690234 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:556                  // 000000001D28: DC69022C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:500                  // 000000001D34: DC6901F4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:492                  // 000000001D40: DC6901EC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:488                  // 000000001D4C: DC6901E8 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:484                  // 000000001D58: DC6901E4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:684                  // 000000001D64: DC6902AC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:632                  // 000000001D70: DC690278 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:628                  // 000000001D7C: DC690274 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:676                  // 000000001D8C: DC6902A4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:672                  // 000000001D9C: DC6902A0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:668                  // 000000001DAC: DC69029C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:664                  // 000000001DBC: DC690298 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:688                  // 000000001DCC: DC6902B0 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:532                  // 000000001DD8: DC690214 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:692                  // 000000001DE4: DC6902B4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:520                  // 000000001DF0: DC690208 007C0000 
 <unknown>
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000001DFC: 916A007E 
	s_cbranch_vccnz 556                                        // 000000001E00: BFA4022C  <_amdgpu_cs_main+0x26B4>
	s_clause 0xd                                               // 000000001E04: BF85000D 
	scratch_load_b128 v[128:131], off, off offset:32           // 000000001E08: DC5D0020 807C0000 
	scratch_load_b128 v[132:135], off, off offset:48           // 000000001E10: DC5D0030 847C0000 
	scratch_load_b128 v[89:92], off, off offset:160            // 000000001E18: DC5D00A0 597C0000 
	scratch_load_b128 v[93:96], off, off offset:176            // 000000001E20: DC5D00B0 5D7C0000 
	scratch_load_b128 v[160:163], off, off offset:352          // 000000001E28: DC5D0160 A07C0000 
	scratch_load_b128 v[164:167], off, off offset:368          // 000000001E30: DC5D0170 A47C0000 
	scratch_load_b128 v[0:3], off, off offset:420              // 000000001E38: DC5D01A4 007C0000 
	scratch_load_b128 v[4:7], off, off offset:436              // 000000001E40: DC5D01B4 047C0000 
	scratch_load_b128 v[168:171], off, off offset:288          // 000000001E48: DC5D0120 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:304          // 000000001E50: DC5D0130 AC7C0000 
	scratch_load_b128 v[24:27], off, off offset:320            // 000000001E58: DC5D0140 187C0000 
	scratch_load_b128 v[28:31], off, off offset:336            // 000000001E60: DC5D0150 1C7C0000 
	scratch_load_b128 v[72:75], off, off offset:64             // 000000001E68: DC5D0040 487C0000 
	scratch_load_b128 v[76:79], off, off offset:80             // 000000001E70: DC5D0050 4C7C0000 
	v_dual_mov_b32 v183, v144 :: v_dual_mov_b32 v182, v143     // 000000001E78: CA100190 B7B6018F 
	v_dual_mov_b32 v181, v142 :: v_dual_mov_b32 v178, v139     // 000000001E80: CA10018E B5B2018B 
	v_dual_mov_b32 v179, v140 :: v_dual_mov_b32 v176, v137     // 000000001E88: CA10018C B3B00189 
	v_mov_b32_e32 v177, v138                                   // 000000001E90: 7F62038A 
	s_clause 0x3                                               // 000000001E94: BF850003 
	scratch_load_b128 v[200:203], off, off offset:96           // 000000001E98: DC5D0060 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:112          // 000000001EA0: DC5D0070 CC7C0000 
	scratch_load_b128 v[145:148], off, off offset:192          // 000000001EA8: DC5D00C0 917C0000 
	scratch_load_b128 v[149:152], off, off offset:208          // 000000001EB0: DC5D00D0 957C0000 
	v_dual_mov_b32 v192, v232 :: v_dual_mov_b32 v193, v233     // 000000001EB8: CA1001E8 C0C001E9 
	v_dual_mov_b32 v194, v234 :: v_dual_mov_b32 v195, v235     // 000000001EC0: CA1001EA C2C201EB 
	v_dual_mov_b32 v196, v236 :: v_dual_mov_b32 v197, v237     // 000000001EC8: CA1001EC C4C401ED 
	v_dual_mov_b32 v198, v238 :: v_dual_mov_b32 v199, v239     // 000000001ED0: CA1001EE C6C601EF 
	s_clause 0x1                                               // 000000001ED8: BF850001 
	scratch_load_b128 v[232:235], off, off offset:256          // 000000001EDC: DC5D0100 E87C0000 
	scratch_load_b128 v[236:239], off, off offset:272          // 000000001EE4: DC5D0110 EC7C0000 
	v_dual_mov_b32 v62, v196 :: v_dual_mov_b32 v63, v197       // 000000001EEC: CA1001C4 3E3E01C5 
	v_dual_mov_b32 v64, v198 :: v_dual_mov_b32 v65, v199       // 000000001EF4: CA1001C6 404001C7 
	v_dual_mov_b32 v61, v195 :: v_dual_mov_b32 v60, v194       // 000000001EFC: CA1001C3 3D3C01C2 
	v_dual_mov_b32 v59, v193 :: v_dual_mov_b32 v58, v192       // 000000001F04: CA1001C1 3B3A01C0 
	v_dual_mov_b32 v112, v223 :: v_dual_mov_b32 v111, v222     // 000000001F0C: CA1001DF 706E01DE 
	v_dual_mov_b32 v110, v221 :: v_dual_mov_b32 v109, v220     // 000000001F14: CA1001DD 6E6C01DC 
	v_dual_mov_b32 v108, v219 :: v_dual_mov_b32 v107, v218     // 000000001F1C: CA1001DB 6C6A01DA 
	v_dual_mov_b32 v106, v217 :: v_dual_mov_b32 v105, v216     // 000000001F24: CA1001D9 6A6801D8 
	v_dual_mov_b32 v120, v191 :: v_dual_mov_b32 v119, v190     // 000000001F2C: CA1001BF 787601BE 
	v_dual_mov_b32 v118, v189 :: v_dual_mov_b32 v117, v188     // 000000001F34: CA1001BD 767401BC 
	v_dual_mov_b32 v116, v187 :: v_dual_mov_b32 v115, v186     // 000000001F3C: CA1001BB 747201BA 
	v_dual_mov_b32 v114, v185 :: v_dual_mov_b32 v113, v184     // 000000001F44: CA1001B9 727001B8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_mov_b32_e32 v180, v141                                   // 000000001F84: 7F68038D 
	s_clause 0x1                                               // 000000001F88: BF850001 
	scratch_load_b128 v[137:140], off, off offset:388          // 000000001F8C: DC5D0184 897C0000 
	scratch_load_b128 v[141:144], off, off offset:404          // 000000001F94: DC5D0194 8D7C0000 
	s_waitcnt vmcnt(21)                                        // 000000001F9C: BF8957F7 
	v_dual_mov_b32 v19, v131 :: v_dual_mov_b32 v16, v128       // 000000001FA0: CA100183 13100180 
	s_waitcnt vmcnt(14)                                        // 000000001FA8: BF893BF7 
	v_wmma_f16_16x16x16_f16 v[89:96], v[160:167], v[0:7], v[89:96]// 000000001FAC: CC424059 1D6601A0 
	s_waitcnt vmcnt(10)                                        // 000000001FB4: BF892BF7 
	v_wmma_f16_16x16x16_f16 v[168:175], v[24:31], v[0:7], v[168:175]// 000000001FB8: CC4240A8 1EA20118 
	s_clause 0x1                                               // 000000001FC0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:128              // 000000001FC4: DC5D0080 007C0000 
	scratch_load_b128 v[4:7], off, off offset:144              // 000000001FCC: DC5D0090 047C0000 
	s_waitcnt vmcnt(11)                                        // 000000001FD4: BF892FF7 
	v_dual_mov_b32 v209, v73 :: v_dual_mov_b32 v208, v72       // 000000001FD8: CA100149 D1D00148 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000001FF0: BF850001 
	scratch_load_b128 v[66:69], off, off                       // 000000001FF4: DC5D0000 427C0000 
	scratch_load_b128 v[70:73], off, off offset:16             // 000000001FFC: DC5D0010 467C0000 
	v_dual_mov_b32 v18, v130 :: v_dual_mov_b32 v17, v129       // 000000002004: CA100182 12100181 
 <unknown>
 <unknown>
	v_dual_mov_b32 v23, v135 :: v_dual_mov_b32 v22, v134       // 000000002014: CA100187 17160186 
	v_dual_mov_b32 v21, v133 :: v_dual_mov_b32 v20, v132       // 00000000201C: CA100185 15140184 
	s_waitcnt vmcnt(11)                                        // 000000002024: BF892FF7 
	v_dual_mov_b32 v53, v203 :: v_dual_mov_b32 v52, v202       // 000000002028: CA1001CB 353401CA 
	s_waitcnt vmcnt(8)                                         // 000000002030: BF8923F7 
	v_dual_mov_b32 v104, v152 :: v_dual_mov_b32 v103, v151     // 000000002034: CA100198 68660197 
	s_waitcnt vmcnt(7)                                         // 00000000203C: BF891FF7 
	v_dual_mov_b32 v84, v235 :: v_dual_mov_b32 v83, v234       // 000000002040: CA1001EB 545201EA 
	s_waitcnt vmcnt(6)                                         // 000000002048: BF891BF7 
	v_dual_mov_b32 v88, v239 :: v_dual_mov_b32 v87, v238       // 00000000204C: CA1001EF 585601EE 
	v_dual_mov_b32 v86, v237 :: v_dual_mov_b32 v85, v236       // 000000002054: CA1001ED 565401EC 
	v_dual_mov_b32 v82, v233 :: v_dual_mov_b32 v81, v232       // 00000000205C: CA1001E9 525001E8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v102, v150 :: v_dual_mov_b32 v101, v149     // 00000000208C: CA100196 66640195 
	v_dual_mov_b32 v100, v148 :: v_dual_mov_b32 v99, v147      // 000000002094: CA100194 64620193 
	v_dual_mov_b32 v98, v146 :: v_dual_mov_b32 v97, v145       // 00000000209C: CA100192 62600191 
	v_dual_mov_b32 v215, v79 :: v_dual_mov_b32 v214, v78       // 0000000020A4: CA10014F D7D6014E 
	v_dual_mov_b32 v213, v77 :: v_dual_mov_b32 v212, v76       // 0000000020AC: CA10014D D5D4014C 
	v_dual_mov_b32 v211, v75 :: v_dual_mov_b32 v210, v74       // 0000000020B4: CA10014B D3D2014A 
	v_dual_mov_b32 v57, v207 :: v_dual_mov_b32 v56, v206       // 0000000020BC: CA1001CF 393801CE 
	v_dual_mov_b32 v55, v205 :: v_dual_mov_b32 v54, v204       // 0000000020C4: CA1001CD 373601CC 
	v_dual_mov_b32 v51, v201 :: v_dual_mov_b32 v50, v200       // 0000000020CC: CA1001C9 333201C8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 00000000215C: BF8903F7 
 <unknown>
	v_wmma_f16_16x16x16_f16 v[89:96], v[160:167], v[137:144], v[89:96] op_sel:[0,0,1]// 000000002164: CC426059 1D6713A0 
	v_wmma_f16_16x16x16_f16 v[168:175], v[24:31], v[137:144], v[168:175] op_sel:[0,0,1]// 00000000216C: CC4260A8 1EA31318 
	s_clause 0x1                                               // 000000002174: BF850001 
	scratch_load_b128 v[137:140], off, off offset:224          // 000000002178: DC5D00E0 897C0000 
	scratch_load_b128 v[141:144], off, off offset:240          // 000000002180: DC5D00F0 8D7C0000 
	v_dual_mov_b32 v24, v176 :: v_dual_mov_b32 v25, v177       // 000000002188: CA1001B0 181801B1 
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 000000002190: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 000000002198: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 0000000021A0: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 0000000021A8: CA100121 A1A00120 
	v_dual_mov_b32 v30, v182 :: v_dual_mov_b32 v31, v183       // 0000000021B0: CA1001B6 1E1E01B7 
	v_dual_mov_b32 v26, v178 :: v_dual_mov_b32 v27, v179       // 0000000021B8: CA1001B2 1A1A01B3 
	v_dual_mov_b32 v28, v180 :: v_dual_mov_b32 v29, v181       // 0000000021C0: CA1001B4 1C1C01B5 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v43, v25 :: v_dual_mov_b32 v42, v24         // 0000000021D8: CA100119 2B2A0118 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v49, v31 :: v_dual_mov_b32 v48, v30         // 000000002220: CA10011F 3130011E 
	v_dual_mov_b32 v47, v29 :: v_dual_mov_b32 v46, v28         // 000000002228: CA10011D 2F2E011C 
	v_dual_mov_b32 v45, v27 :: v_dual_mov_b32 v44, v26         // 000000002230: CA10011B 2D2C011A 
 <unknown>
	scratch_store_b32 off, v128, off offset:384                // 00000000223C: DC690180 007C8000 
	v_dual_mov_b32 v249, v1 :: v_dual_mov_b32 v248, v0         // 000000002244: CA100101 F9F80100 
 <unknown>
	v_dual_mov_b32 v255, v7 :: v_dual_mov_b32 v254, v6         // 000000002250: CA100107 FFFE0106 
	v_dual_mov_b32 v253, v5 :: v_dual_mov_b32 v252, v4         // 000000002258: CA100105 FDFC0104 
	scratch_store_b32 off, v0, off offset:716                  // 000000002260: DC6902CC 007C0000 
 <unknown>
	v_dual_mov_b32 v251, v3 :: v_dual_mov_b32 v250, v2         // 00000000226C: CA100103 FBFA0102 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:712                  // 000000002280: DC6902C8 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:624                  // 0000000022A4: DC690270 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:620                  // 0000000022D0: DC69026C 007C0000 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:708                  // 0000000022E4: DC6902C4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:704                  // 0000000022F0: DC6902C0 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:700                  // 0000000022FC: DC6902BC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:696                  // 000000002308: DC6902B8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:748                  // 000000002318: DC6902EC 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:744                  // 000000002328: DC6902E8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:740                  // 000000002338: DC6902E4 007C0000 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 000000002348: BF8907F7 
	v_dual_mov_b32 v243, v140 :: v_dual_mov_b32 v242, v139     // 00000000234C: CA10018C F3F2018B 
	scratch_store_b32 off, v0, off offset:736                  // 000000002354: DC6902E0 007C0000 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 000000002364: BF8903F7 
	v_dual_mov_b32 v247, v144 :: v_dual_mov_b32 v246, v143     // 000000002368: CA100190 F7F6018F 
	v_dual_mov_b32 v245, v142 :: v_dual_mov_b32 v244, v141     // 000000002370: CA10018E F5F4018D 
	scratch_store_b32 off, v0, off offset:732                  // 000000002378: DC6902DC 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v241, v138 :: v_dual_mov_b32 v240, v137     // 000000002388: CA10018A F1F00189 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:728                  // 0000000023A0: DC6902D8 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:724                  // 0000000023B8: DC6902D4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:720                  // 0000000023C8: DC6902D0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:560                  // 0000000023D8: DC690230 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:564                  // 0000000023E8: DC690234 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:556                  // 0000000023F8: DC69022C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:676                  // 000000002408: DC6902A4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:672                  // 000000002418: DC6902A0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:668                  // 000000002428: DC69029C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:664                  // 000000002438: DC690298 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:688                  // 000000002448: DC6902B0 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:532                  // 000000002454: DC690214 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:692                  // 000000002460: DC6902B4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:520                  // 00000000246C: DC690208 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:540                  // 000000002478: DC69021C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:528                  // 000000002484: DC690210 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:524                  // 000000002490: DC69020C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:544                  // 00000000249C: DC690220 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:512                  // 0000000024A8: DC690200 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:508                  // 0000000024B4: DC6901FC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:504                  // 0000000024C0: DC6901F8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:648                  // 0000000024D0: DC690288 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 0000000024E0: DC690228 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:644                  // 0000000024F0: DC690284 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:680                  // 000000002500: DC6902A8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:496                  // 000000002510: DC6901F0 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v239, v199 :: v_dual_mov_b32 v238, v198     // 000000002520: CA1001C7 EFEE01C6 
	v_dual_mov_b32 v237, v197 :: v_dual_mov_b32 v236, v196     // 000000002528: CA1001C5 EDEC01C4 
	scratch_store_b32 off, v0, off offset:480                  // 000000002530: DC6901E0 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v235, v195 :: v_dual_mov_b32 v234, v194     // 000000002540: CA1001C3 EBEA01C2 
	v_dual_mov_b32 v233, v193 :: v_dual_mov_b32 v232, v192     // 000000002548: CA1001C1 E9E801C0 
	scratch_store_b32 off, v0, off offset:640                  // 000000002550: DC690280 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:636                  // 000000002560: DC69027C 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:548                  // 000000002570: DC690224 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:476                  // 000000002580: DC6901DC 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v144, v31 :: v_dual_mov_b32 v143, v30       // 000000002590: CA10011F 908E011E 
	v_dual_mov_b32 v142, v29 :: v_dual_mov_b32 v141, v28       // 000000002598: CA10011D 8E8C011C 
	scratch_store_b32 off, v0, off offset:472                  // 0000000025A0: DC6901D8 007C0000 
 <unknown>
	v_dual_mov_b32 v140, v27 :: v_dual_mov_b32 v139, v26       // 0000000025AC: CA10011B 8C8A011A 
	v_dual_mov_b32 v138, v25 :: v_dual_mov_b32 v137, v24       // 0000000025B4: CA100119 8A880118 
	scratch_store_b32 off, v0, off offset:468                  // 0000000025BC: DC6901D4 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:464                  // 0000000025D4: DC6901D0 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:536                 // 0000000025E0: DC690218 007C1900 
 <unknown>
	scratch_store_b32 off, v0, off offset:460                  // 0000000025EC: DC6901CC 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:516                 // 0000000025F8: DC690204 007C1900 
 <unknown>
	scratch_store_b32 off, v0, off offset:456                  // 000000002604: DC6901C8 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:652                 // 000000002610: DC69028C 007C1900 
 <unknown>
	scratch_store_b32 off, v0, off offset:452                  // 00000000261C: DC6901C4 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:656                 // 000000002628: DC690290 007C1900 
 <unknown>
	scratch_store_b32 off, v0, off offset:500                  // 000000002634: DC6901F4 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:660                 // 000000002640: DC690294 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:492                  // 000000002650: DC6901EC 007C0000 
 <unknown>
	scratch_store_b32 off, v25, off offset:576                 // 00000000265C: DC690240 007C1900 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:488                  // 00000000266C: DC6901E8 007C0000 
 <unknown>
	s_clause 0x1                                               // 000000002678: BF850001 
	scratch_store_b32 off, v25, off offset:568                 // 00000000267C: DC690238 007C1900 
	scratch_store_b32 off, v0, off offset:484                  // 000000002684: DC6901E4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:684                  // 000000002690: DC6902AC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:632                  // 00000000269C: DC690278 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:628                  // 0000000026A8: DC690274 007C0000 
 <unknown>
	s_clause 0x3                                               // 0000000026B4: BF850003 
	scratch_load_b32 v26, off, off offset:636                  // 0000000026B8: DC51027C 1A7C0000 
	scratch_load_b32 v27, off, off offset:640                  // 0000000026C0: DC510280 1B7C0000 
	scratch_load_b32 v28, off, off offset:680                  // 0000000026C8: DC5102A8 1C7C0000 
	scratch_load_b32 v68, off, off offset:684                  // 0000000026D0: DC5102AC 447C0000 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 0000000026E0: BF850001 
	scratch_store_b32 off, v135, off offset:572                // 0000000026E4: DC69023C 007C8700 
	scratch_store_b32 off, v24, off offset:616                 // 0000000026EC: DC690268 007C1800 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x9                                               // 000000002710: BF850009 
	scratch_load_b32 v194, off, off offset:704                 // 000000002714: DC5102C0 C27C0000 
	scratch_load_b32 v195, off, off offset:708                 // 00000000271C: DC5102C4 C37C0000 
	scratch_load_b32 v198, off, off offset:712                 // 000000002724: DC5102C8 C67C0000 
	scratch_load_b32 v40, off, off offset:656                  // 00000000272C: DC510290 287C0000 
	scratch_load_b32 v130, off, off offset:652                 // 000000002734: DC51028C 827C0000 
	scratch_load_b32 v225, off, off offset:648                 // 00000000273C: DC510288 E17C0000 
	scratch_load_b32 v128, off, off offset:628                 // 000000002744: DC510274 807C0000 
	scratch_load_b32 v24, off, off offset:632                  // 00000000274C: DC510278 187C0000 
	scratch_load_b32 v30, off, off offset:660                  // 000000002754: DC510294 1E7C0000 
	scratch_load_b32 v31, off, off offset:644                  // 00000000275C: DC510284 1F7C0000 
	s_mov_b32 s7, 0                                            // 000000002764: BE870080 
	s_waitcnt vmcnt(10)                                        // 000000002768: BF892BF7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x7                                               // 00000000277C: BF850007 
	scratch_load_b32 v132, off, off offset:692                 // 000000002780: DC5102B4 847C0000 
	scratch_load_b32 v201, off, off offset:688                 // 000000002788: DC5102B0 C97C0000 
	scratch_load_b32 v29, off, off offset:664                  // 000000002790: DC510298 1D7C0000 
	scratch_load_b32 v121, off, off offset:668                 // 000000002798: DC51029C 797C0000 
	scratch_load_b32 v122, off, off offset:672                 // 0000000027A0: DC5102A0 7A7C0000 
	scratch_load_b32 v123, off, off offset:676                 // 0000000027A8: DC5102A4 7B7C0000 
	scratch_load_b32 v135, off, off offset:696                 // 0000000027B0: DC5102B8 877C0000 
	scratch_load_b32 v193, off, off offset:700                 // 0000000027B8: DC5102BC C17C0000 
	s_waitcnt vmcnt(17)                                        // 0000000027C0: BF8947F7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(16)                                        // 0000000027CC: BF8943F7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(15)                                        // 0000000027D8: BF893FF7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 0000000027E4: BF8903F7 
 <unknown>
 <unknown>
	s_clause 0x8                                               // 0000000027F0: BF850008 
	scratch_load_b32 v179, off, off offset:716                 // 0000000027F4: DC5102CC B37C0000 
	scratch_load_b32 v192, off, off offset:720                 // 0000000027FC: DC5102D0 C07C0000 
	scratch_load_b32 v193, off, off offset:724                 // 000000002804: DC5102D4 C17C0000 
	scratch_load_b32 v194, off, off offset:728                 // 00000000280C: DC5102D8 C27C0000 
	scratch_load_b32 v195, off, off offset:732                 // 000000002814: DC5102DC C37C0000 
	scratch_load_b32 v196, off, off offset:736                 // 00000000281C: DC5102E0 C47C0000 
	scratch_load_b32 v197, off, off offset:740                 // 000000002824: DC5102E4 C57C0000 
	scratch_load_b32 v198, off, off offset:744                 // 00000000282C: DC5102E8 C67C0000 
	scratch_load_b32 v199, off, off offset:748                 // 000000002834: DC5102EC C77C0000 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 00000000283C: 916A077E 
	s_cbranch_vccnz 514                                        // 000000002840: BFA40202  <_amdgpu_cs_main+0x304C>
	s_clause 0xb                                               // 000000002844: BF85000B 
	scratch_load_b128 v[0:3], off, off offset:32               // 000000002848: DC5D0020 007C0000 
	scratch_load_b128 v[4:7], off, off offset:48               // 000000002850: DC5D0030 047C0000 
	scratch_load_b128 v[72:75], off, off offset:64             // 000000002858: DC5D0040 487C0000 
	scratch_load_b128 v[76:79], off, off offset:80             // 000000002860: DC5D0050 4C7C0000 
	scratch_load_b128 v[81:84], off, off offset:256            // 000000002868: DC5D0100 517C0000 
	scratch_load_b128 v[85:88], off, off offset:272            // 000000002870: DC5D0110 557C0000 
	scratch_load_b128 v[160:163], off, off offset:352          // 000000002878: DC5D0160 A07C0000 
	scratch_load_b128 v[164:167], off, off offset:368          // 000000002880: DC5D0170 A47C0000 
	scratch_load_b128 v[40:43], off, off offset:420            // 000000002888: DC5D01A4 287C0000 
	scratch_load_b128 v[44:47], off, off offset:436            // 000000002890: DC5D01B4 2C7C0000 
	scratch_load_b128 v[24:27], off, off offset:320            // 000000002898: DC5D0140 187C0000 
	scratch_load_b128 v[28:31], off, off offset:336            // 0000000028A0: DC5D0150 1C7C0000 
	v_dual_mov_b32 v175, v144 :: v_dual_mov_b32 v174, v143     // 0000000028A8: CA100190 AFAE018F 
	v_dual_mov_b32 v173, v142 :: v_dual_mov_b32 v170, v139     // 0000000028B0: CA10018E ADAA018B 
	v_dual_mov_b32 v171, v140 :: v_dual_mov_b32 v168, v137     // 0000000028B8: CA10018C ABA80189 
	v_mov_b32_e32 v169, v138                                   // 0000000028C0: 7F52038A 
	s_clause 0x5                                               // 0000000028C4: BF850005 
	scratch_load_b128 v[145:148], off, off offset:288          // 0000000028C8: DC5D0120 917C0000 
	scratch_load_b128 v[149:152], off, off offset:304          // 0000000028D0: DC5D0130 957C0000 
	scratch_load_b128 v[121:124], off, off offset:160          // 0000000028D8: DC5D00A0 797C0000 
	scratch_load_b128 v[125:128], off, off offset:176          // 0000000028E0: DC5D00B0 7D7C0000 
	scratch_load_b128 v[129:132], off, off offset:192          // 0000000028E8: DC5D00C0 817C0000 
	scratch_load_b128 v[133:136], off, off offset:208          // 0000000028F0: DC5D00D0 857C0000 
	v_dual_mov_b32 v113, v184 :: v_dual_mov_b32 v114, v185     // 0000000028F8: CA1001B8 717201B9 
	v_dual_mov_b32 v115, v186 :: v_dual_mov_b32 v116, v187     // 000000002900: CA1001BA 737401BB 
	v_dual_mov_b32 v117, v188 :: v_dual_mov_b32 v118, v189     // 000000002908: CA1001BC 757601BD 
	v_dual_mov_b32 v119, v190 :: v_dual_mov_b32 v120, v191     // 000000002910: CA1001BE 777801BF 
	s_clause 0x1                                               // 000000002918: BF850001 
	scratch_store_b128 off, v[184:187], off offset:584         // 00000000291C: DC750248 007CB800 
	scratch_store_b128 off, v[188:191], off offset:600         // 000000002924: DC750258 007CBC00 
	s_clause 0x3                                               // 00000000292C: BF850003 
	scratch_load_b128 v[184:187], off, off offset:128          // 000000002930: DC5D0080 B87C0000 
	scratch_load_b128 v[188:191], off, off offset:144          // 000000002938: DC5D0090 BC7C0000 
	scratch_load_b128 v[200:203], off, off offset:96           // 000000002940: DC5D0060 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:112          // 000000002948: DC5D0070 CC7C0000 
	v_dual_mov_b32 v65, v239 :: v_dual_mov_b32 v64, v238       // 000000002950: CA1001EF 414001EE 
	v_dual_mov_b32 v63, v237 :: v_dual_mov_b32 v62, v236       // 000000002958: CA1001ED 3F3E01EC 
	v_dual_mov_b32 v61, v235 :: v_dual_mov_b32 v60, v234       // 000000002960: CA1001EB 3D3C01EA 
	v_dual_mov_b32 v59, v233 :: v_dual_mov_b32 v58, v232       // 000000002968: CA1001E9 3B3A01E8 
	v_dual_mov_b32 v112, v223 :: v_dual_mov_b32 v111, v222     // 000000002970: CA1001DF 706E01DE 
	v_dual_mov_b32 v110, v221 :: v_dual_mov_b32 v109, v220     // 000000002978: CA1001DD 6E6C01DC 
	v_dual_mov_b32 v108, v219 :: v_dual_mov_b32 v107, v218     // 000000002980: CA1001DB 6C6A01DA 
	v_dual_mov_b32 v106, v217 :: v_dual_mov_b32 v105, v216     // 000000002988: CA1001D9 6A6801D8 
	s_waitcnt vmcnt(30)                                        // 000000002990: BF897BF7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_mov_b32_e32 v172, v141                                   // 0000000029C4: 7F58038D 
	s_clause 0x1                                               // 0000000029C8: BF850001 
	scratch_load_b128 v[137:140], off, off offset:388          // 0000000029CC: DC5D0184 897C0000 
	scratch_load_b128 v[141:144], off, off offset:404          // 0000000029D4: DC5D0194 8D7C0000 
	s_waitcnt vmcnt(23)                                        // 0000000029DC: BF895FF7 
	v_mov_b32_e32 v19, v3                                      // 0000000029E0: 7E260303 
	s_waitcnt vmcnt(21)                                        // 0000000029E4: BF8957F7 
	v_dual_mov_b32 v209, v73 :: v_dual_mov_b32 v208, v72       // 0000000029E8: CA100149 D1D00148 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000002A00: BF850001 
	scratch_load_b128 v[66:69], off, off                       // 000000002A04: DC5D0000 427C0000 
	scratch_load_b128 v[70:73], off, off offset:16             // 000000002A0C: DC5D0010 467C0000 
	s_waitcnt vmcnt(16)                                        // 000000002A14: BF8943F7 
	v_wmma_f16_16x16x16_f16 v[81:88], v[160:167], v[40:47], v[81:88]// 000000002A18: CC424051 1D4651A0 
	s_waitcnt vmcnt(14)                                        // 000000002A20: BF893BF7 
	v_wmma_f16_16x16x16_f16 v[113:120], v[24:31], v[40:47], v[113:120]// 000000002A24: CC424071 1DC65118 
	v_mov_b32_e32 v16, v0                                      // 000000002A2C: 7E200300 
 <unknown>
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6           // 000000002A34: CA100107 17160106 
	s_waitcnt vmcnt(10)                                        // 000000002A3C: BF892BF7 
	v_dual_mov_b32 v94, v126 :: v_dual_mov_b32 v93, v125       // 000000002A40: CA10017E 5E5C017D 
	scratch_store_b32 off, v0, off offset:624                  // 000000002A48: DC690270 007C0000 
 <unknown>
	s_waitcnt vmcnt(9)                                         // 000000002A54: BF8927F7 
	v_dual_mov_b32 v98, v130 :: v_dual_mov_b32 v97, v129       // 000000002A58: CA100182 62600181 
	v_dual_mov_b32 v96, v128 :: v_dual_mov_b32 v95, v127       // 000000002A60: CA100180 605E017F 
	scratch_store_b32 off, v0, off offset:620                  // 000000002A68: DC69026C 007C0000 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(7)                                         // 000000002A78: BF891FF7 
	v_dual_mov_b32 v251, v187 :: v_dual_mov_b32 v250, v186     // 000000002A7C: CA1001BB FBFA01BA 
	s_waitcnt vmcnt(5)                                         // 000000002A84: BF8917F7 
	v_dual_mov_b32 v51, v201 :: v_dual_mov_b32 v50, v200       // 000000002A88: CA1001C9 333201C8 
	scratch_store_b32 off, v0, off offset:576                  // 000000002A90: DC690240 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v255, v191 :: v_dual_mov_b32 v254, v190     // 000000002AA0: CA1001BF FFFE01BE 
	v_dual_mov_b32 v253, v189 :: v_dual_mov_b32 v252, v188     // 000000002AA8: CA1001BD FDFC01BC 
	scratch_store_b32 off, v0, off offset:568                  // 000000002AB0: DC690238 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v249, v185 :: v_dual_mov_b32 v248, v184     // 000000002AC0: CA1001B9 F9F801B8 
	s_waitcnt vmcnt(0)                                         // 000000002AC8: BF8903F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:560                  // 000000002ADC: DC690230 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4           // 000000002AFC: CA100105 15140104 
	scratch_store_b32 off, v0, off offset:564                  // 000000002B04: DC690234 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1           // 000000002B14: CA100102 12100101 
	v_dual_mov_b32 v92, v124 :: v_dual_mov_b32 v91, v123       // 000000002B1C: CA10017C 5C5A017B 
	scratch_store_b32 off, v0, off offset:556                  // 000000002B24: DC69022C 007C0000 
 <unknown>
	v_dual_mov_b32 v90, v122 :: v_dual_mov_b32 v89, v121       // 000000002B30: CA10017A 5A580179 
	v_dual_mov_b32 v104, v136 :: v_dual_mov_b32 v103, v135     // 000000002B38: CA100188 68660187 
	scratch_store_b32 off, v0, off offset:532                  // 000000002B40: DC690214 007C0000 
 <unknown>
	v_dual_mov_b32 v102, v134 :: v_dual_mov_b32 v101, v133     // 000000002B4C: CA100186 66640185 
	v_dual_mov_b32 v100, v132 :: v_dual_mov_b32 v99, v131      // 000000002B54: CA100184 64620183 
	scratch_store_b32 off, v0, off offset:520                  // 000000002B5C: DC690208 007C0000 
	v_dual_mov_b32 v215, v79 :: v_dual_mov_b32 v214, v78       // 000000002B64: CA10014F D7D6014E 
	v_dual_mov_b32 v213, v77 :: v_dual_mov_b32 v212, v76       // 000000002B6C: CA10014D D5D4014C 
	v_dual_mov_b32 v211, v75 :: v_dual_mov_b32 v210, v74       // 000000002B74: CA10014B D3D2014A 
	v_dual_mov_b32 v57, v207 :: v_dual_mov_b32 v56, v206       // 000000002B7C: CA1001CF 393801CE 
	v_dual_mov_b32 v55, v205 :: v_dual_mov_b32 v54, v204       // 000000002B84: CA1001CD 373601CC 
	v_dual_mov_b32 v53, v203 :: v_dual_mov_b32 v52, v202       // 000000002B8C: CA1001CB 353401CA 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_wmma_f16_16x16x16_f16 v[81:88], v[160:167], v[137:144], v[81:88] op_sel:[0,0,1]// 000000002CC8: CC426051 1D4713A0 
	v_wmma_f16_16x16x16_f16 v[113:120], v[24:31], v[137:144], v[113:120] op_sel:[0,0,1]// 000000002CD0: CC426071 1DC71318 
	s_clause 0x1                                               // 000000002CD8: BF850001 
	scratch_load_b128 v[137:140], off, off offset:224          // 000000002CDC: DC5D00E0 897C0000 
	scratch_load_b128 v[141:144], off, off offset:240          // 000000002CE4: DC5D00F0 8D7C0000 
	v_dual_mov_b32 v24, v168 :: v_dual_mov_b32 v25, v169       // 000000002CEC: CA1001A8 181801A9 
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 000000002CF4: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 000000002CFC: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 000000002D04: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 000000002D0C: CA100121 A1A00120 
	v_dual_mov_b32 v30, v174 :: v_dual_mov_b32 v31, v175       // 000000002D14: CA1001AE 1E1E01AF 
	v_dual_mov_b32 v26, v170 :: v_dual_mov_b32 v27, v171       // 000000002D1C: CA1001AA 1A1A01AB 
	v_dual_mov_b32 v28, v172 :: v_dual_mov_b32 v29, v173       // 000000002D24: CA1001AC 1C1C01AD 
	v_dual_mov_b32 v175, v152 :: v_dual_mov_b32 v174, v151     // 000000002D2C: CA100198 AFAE0197 
	v_dual_mov_b32 v173, v150 :: v_dual_mov_b32 v172, v149     // 000000002D34: CA100196 ADAC0195 
	v_dual_mov_b32 v171, v148 :: v_dual_mov_b32 v170, v147     // 000000002D3C: CA100194 ABAA0193 
	v_dual_mov_b32 v169, v146 :: v_dual_mov_b32 v168, v145     // 000000002D44: CA100192 A9A80191 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v43, v25 :: v_dual_mov_b32 v42, v24         // 000000002D80: CA100119 2B2A0118 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v47, v29 :: v_dual_mov_b32 v46, v28         // 000000002DEC: CA10011D 2F2E011C 
	v_dual_mov_b32 v49, v31 :: v_dual_mov_b32 v48, v30         // 000000002DF4: CA10011F 3130011E 
	v_dual_mov_b32 v45, v27 :: v_dual_mov_b32 v44, v26         // 000000002DFC: CA10011B 2D2C011A 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v128, off offset:384                // 000000002E10: DC690180 007C8000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v126, off offset:572                // 000000002E20: DC69023C 007C7E00 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:540                  // 000000002E34: DC69021C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:528                  // 000000002E40: DC690210 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:524                  // 000000002E4C: DC69020C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:544                  // 000000002E58: DC690220 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:512                  // 000000002E64: DC690200 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:508                  // 000000002E70: DC6901FC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:504                  // 000000002E7C: DC6901F8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 000000002E8C: DC690228 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:536                  // 000000002E98: DC690218 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 000000002EA4: DC690204 007C0000 
	s_waitcnt vmcnt(1)                                         // 000000002EAC: BF8907F7 
	v_dual_mov_b32 v243, v140 :: v_dual_mov_b32 v242, v139     // 000000002EB0: CA10018C F3F2018B 
	s_waitcnt vmcnt(0)                                         // 000000002EB8: BF8903F7 
 <unknown>
 <unknown>
	v_dual_mov_b32 v247, v144 :: v_dual_mov_b32 v246, v143     // 000000002EC4: CA100190 F7F6018F 
	v_dual_mov_b32 v245, v142 :: v_dual_mov_b32 v244, v141     // 000000002ECC: CA10018E F5F4018D 
	scratch_store_b32 off, v0, off offset:548                  // 000000002ED4: DC690224 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v241, v138 :: v_dual_mov_b32 v240, v137     // 000000002EE4: CA10018A F1F00189 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:476                  // 000000002EFC: DC6901DC 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v144, v31 :: v_dual_mov_b32 v143, v30       // 000000002F24: CA10011F 908E011E 
	scratch_store_b32 off, v0, off offset:472                  // 000000002F2C: DC6901D8 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v142, v29 :: v_dual_mov_b32 v141, v28       // 000000002F3C: CA10011D 8E8C011C 
	v_dual_mov_b32 v140, v27 :: v_dual_mov_b32 v139, v26       // 000000002F44: CA10011B 8C8A011A 
	scratch_store_b32 off, v0, off offset:468                  // 000000002F4C: DC6901D4 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v138, v25 :: v_dual_mov_b32 v137, v24       // 000000002F5C: CA100119 8A880118 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:464                  // 000000002F6C: DC6901D0 007C0000 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:496                 // 000000002F80: DC6901F0 007C1D00 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:460                  // 000000002F90: DC6901CC 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:456                  // 000000002FB8: DC6901C8 007C0000 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000002FC8: BF850001 
	scratch_load_b128 v[184:187], off, off offset:584          // 000000002FCC: DC5D0248 B87C0000 
	scratch_load_b128 v[188:191], off, off offset:600          // 000000002FD4: DC5D0258 BC7C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000002FF0: BF850001 
	scratch_store_b32 off, v0, off offset:452                  // 000000002FF4: DC6901C4 007C0000 
	scratch_store_b32 off, v29, off offset:480                 // 000000002FFC: DC6901E0 007C1D00 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:500                  // 000000003010: DC6901F4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:492                  // 00000000301C: DC6901EC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:488                  // 000000003028: DC6901E8 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:484                  // 000000003034: DC6901E4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:616                  // 000000003040: DC690268 007C0000 
 <unknown>
	s_clause 0x2                                               // 00000000304C: BF850002 
	scratch_load_b32 v153, off, off offset:616                 // 000000003050: DC510268 997C0000 
	scratch_load_b32 v126, off, off offset:620                 // 000000003058: DC51026C 7E7C0000 
	scratch_load_b32 v152, off, off offset:624                 // 000000003060: DC510270 987C0000 
	s_mov_b32 s7, 0                                            // 000000003068: BE870080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000306C: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000003070: 916A077E 
	s_cbranch_vccnz 502                                        // 000000003074: BFA401F6  <_amdgpu_cs_main+0x3850>
	s_clause 0xb                                               // 000000003078: BF85000B 
	scratch_load_b128 v[0:3], off, off offset:32               // 00000000307C: DC5D0020 007C0000 
	scratch_load_b128 v[4:7], off, off offset:48               // 000000003084: DC5D0030 047C0000 
	scratch_load_b128 v[72:75], off, off offset:64             // 00000000308C: DC5D0040 487C0000 
	scratch_load_b128 v[76:79], off, off offset:80             // 000000003094: DC5D0050 4C7C0000 
	scratch_load_b128 v[240:243], off, off offset:224          // 00000000309C: DC5D00E0 F07C0000 
	scratch_load_b128 v[244:247], off, off offset:240          // 0000000030A4: DC5D00F0 F47C0000 
	scratch_load_b128 v[160:163], off, off offset:352          // 0000000030AC: DC5D0160 A07C0000 
	scratch_load_b128 v[164:167], off, off offset:368          // 0000000030B4: DC5D0170 A47C0000 
	scratch_load_b128 v[40:43], off, off offset:420            // 0000000030BC: DC5D01A4 287C0000 
	scratch_load_b128 v[44:47], off, off offset:436            // 0000000030C4: DC5D01B4 2C7C0000 
	scratch_load_b128 v[24:27], off, off offset:320            // 0000000030CC: DC5D0140 187C0000 
	scratch_load_b128 v[28:31], off, off offset:336            // 0000000030D4: DC5D0150 1C7C0000 
	v_dual_mov_b32 v175, v144 :: v_dual_mov_b32 v174, v143     // 0000000030DC: CA100190 AFAE018F 
	v_dual_mov_b32 v173, v142 :: v_dual_mov_b32 v170, v139     // 0000000030E4: CA10018E ADAA018B 
	v_dual_mov_b32 v171, v140 :: v_dual_mov_b32 v168, v137     // 0000000030EC: CA10018C ABA80189 
	v_mov_b32_e32 v169, v138                                   // 0000000030F4: 7F52038A 
	s_clause 0x5                                               // 0000000030F8: BF850005 
	scratch_load_b128 v[145:148], off, off offset:288          // 0000000030FC: DC5D0120 917C0000 
	scratch_load_b128 v[149:152], off, off offset:304          // 000000003104: DC5D0130 957C0000 
	scratch_load_b128 v[121:124], off, off offset:160          // 00000000310C: DC5D00A0 797C0000 
	scratch_load_b128 v[125:128], off, off offset:176          // 000000003114: DC5D00B0 7D7C0000 
	scratch_load_b128 v[129:132], off, off offset:192          // 00000000311C: DC5D00C0 817C0000 
	scratch_load_b128 v[133:136], off, off offset:208          // 000000003124: DC5D00D0 857C0000 
	v_dual_mov_b32 v105, v216 :: v_dual_mov_b32 v106, v217     // 00000000312C: CA1001D8 696A01D9 
	v_dual_mov_b32 v107, v218 :: v_dual_mov_b32 v108, v219     // 000000003134: CA1001DA 6B6C01DB 
	v_dual_mov_b32 v109, v220 :: v_dual_mov_b32 v110, v221     // 00000000313C: CA1001DC 6D6E01DD 
	v_dual_mov_b32 v111, v222 :: v_dual_mov_b32 v112, v223     // 000000003144: CA1001DE 6F7001DF 
	s_clause 0x1                                               // 00000000314C: BF850001 
	scratch_store_b128 off, v[216:219], off offset:584         // 000000003150: DC750248 007CD800 
	scratch_store_b128 off, v[220:223], off offset:600         // 000000003158: DC750258 007CDC00 
	s_clause 0x3                                               // 000000003160: BF850003 
	scratch_load_b128 v[216:219], off, off offset:128          // 000000003164: DC5D0080 D87C0000 
	scratch_load_b128 v[220:223], off, off offset:144          // 00000000316C: DC5D0090 DC7C0000 
	scratch_load_b128 v[200:203], off, off offset:96           // 000000003174: DC5D0060 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:112          // 00000000317C: DC5D0070 CC7C0000 
	v_dual_mov_b32 v65, v239 :: v_dual_mov_b32 v64, v238       // 000000003184: CA1001EF 414001EE 
	v_dual_mov_b32 v63, v237 :: v_dual_mov_b32 v62, v236       // 00000000318C: CA1001ED 3F3E01EC 
	v_dual_mov_b32 v61, v235 :: v_dual_mov_b32 v60, v234       // 000000003194: CA1001EB 3D3C01EA 
	v_dual_mov_b32 v59, v233 :: v_dual_mov_b32 v58, v232       // 00000000319C: CA1001E9 3B3A01E8 
	s_waitcnt vmcnt(25)                                        // 0000000031A4: BF8967F7 
	v_dual_mov_b32 v120, v191 :: v_dual_mov_b32 v119, v190     // 0000000031A8: CA1001BF 787601BE 
	v_dual_mov_b32 v118, v189 :: v_dual_mov_b32 v117, v188     // 0000000031B0: CA1001BD 767401BC 
	v_dual_mov_b32 v116, v187 :: v_dual_mov_b32 v115, v186     // 0000000031B8: CA1001BB 747201BA 
	v_dual_mov_b32 v114, v185 :: v_dual_mov_b32 v113, v184     // 0000000031C0: CA1001B9 727001B8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_mov_b32_e32 v172, v141                                   // 000000003228: 7F58038D 
	s_clause 0x1                                               // 00000000322C: BF850001 
	scratch_load_b128 v[137:140], off, off offset:388          // 000000003230: DC5D0184 897C0000 
	scratch_load_b128 v[141:144], off, off offset:404          // 000000003238: DC5D0194 8D7C0000 
	s_waitcnt vmcnt(23)                                        // 000000003240: BF895FF7 
	v_mov_b32_e32 v19, v3                                      // 000000003244: 7E260303 
	s_waitcnt vmcnt(21)                                        // 000000003248: BF8957F7 
	v_dual_mov_b32 v209, v73 :: v_dual_mov_b32 v208, v72       // 00000000324C: CA100149 D1D00148 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000003264: BF850001 
	scratch_load_b128 v[66:69], off, off                       // 000000003268: DC5D0000 427C0000 
	scratch_load_b128 v[70:73], off, off offset:16             // 000000003270: DC5D0010 467C0000 
	s_waitcnt vmcnt(16)                                        // 000000003278: BF8943F7 
	v_wmma_f16_16x16x16_f16 v[240:247], v[160:167], v[40:47], v[240:247]// 00000000327C: CC4240F0 1FC251A0 
	s_waitcnt vmcnt(14)                                        // 000000003284: BF893BF7 
	v_wmma_f16_16x16x16_f16 v[105:112], v[24:31], v[40:47], v[105:112]// 000000003288: CC424069 1DA65118 
	v_dual_mov_b32 v16, v0 :: v_dual_mov_b32 v23, v7           // 000000003290: CA100100 10160107 
	v_mov_b32_e32 v22, v6                                      // 000000003298: 7E2C0306 
	s_waitcnt vmcnt(10)                                        // 00000000329C: BF892BF7 
	v_dual_mov_b32 v96, v128 :: v_dual_mov_b32 v95, v127       // 0000000032A0: CA100180 605E017F 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(8)                                         // 0000000032B0: BF8923F7 
	v_dual_mov_b32 v102, v134 :: v_dual_mov_b32 v101, v133     // 0000000032B4: CA100186 66640185 
 <unknown>
 <unknown>
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4           // 0000000032C4: CA100105 15140104 
	scratch_store_b32 off, v0, off offset:576                  // 0000000032CC: DC690240 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1           // 0000000032DC: CA100102 12100101 
	s_waitcnt vmcnt(5)                                         // 0000000032E4: BF8917F7 
	v_dual_mov_b32 v53, v203 :: v_dual_mov_b32 v52, v202       // 0000000032E8: CA1001CB 353401CA 
	scratch_store_b32 off, v0, off offset:568                  // 0000000032F0: DC690238 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v255, v223 :: v_dual_mov_b32 v254, v222     // 000000003300: CA1001DF FFFE01DE 
	v_dual_mov_b32 v253, v221 :: v_dual_mov_b32 v252, v220     // 000000003308: CA1001DD FDFC01DC 
	scratch_store_b32 off, v0, off offset:560                  // 000000003310: DC690230 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v251, v219 :: v_dual_mov_b32 v250, v218     // 000000003320: CA1001DB FBFA01DA 
	v_dual_mov_b32 v249, v217 :: v_dual_mov_b32 v248, v216     // 000000003328: CA1001D9 F9F801D8 
	scratch_store_b32 off, v0, off offset:564                  // 000000003330: DC690234 007C0000 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 000000003340: BF8903F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v94, v126 :: v_dual_mov_b32 v93, v125       // 00000000335C: CA10017E 5E5C017D 
	scratch_store_b32 off, v0, off offset:556                  // 000000003364: DC69022C 007C0000 
 <unknown>
	v_dual_mov_b32 v92, v124 :: v_dual_mov_b32 v91, v123       // 000000003370: CA10017C 5C5A017B 
	v_dual_mov_b32 v90, v122 :: v_dual_mov_b32 v89, v121       // 000000003378: CA10017A 5A580179 
	scratch_store_b32 off, v0, off offset:532                  // 000000003380: DC690214 007C0000 
 <unknown>
	v_dual_mov_b32 v104, v136 :: v_dual_mov_b32 v103, v135     // 00000000338C: CA100188 68660187 
	v_dual_mov_b32 v100, v132 :: v_dual_mov_b32 v99, v131      // 000000003394: CA100184 64620183 
	scratch_store_b32 off, v0, off offset:520                  // 00000000339C: DC690208 007C0000 
	v_dual_mov_b32 v98, v130 :: v_dual_mov_b32 v97, v129       // 0000000033A4: CA100182 62600181 
	v_dual_mov_b32 v215, v79 :: v_dual_mov_b32 v214, v78       // 0000000033AC: CA10014F D7D6014E 
	v_dual_mov_b32 v213, v77 :: v_dual_mov_b32 v212, v76       // 0000000033B4: CA10014D D5D4014C 
	v_dual_mov_b32 v211, v75 :: v_dual_mov_b32 v210, v74       // 0000000033BC: CA10014B D3D2014A 
	v_dual_mov_b32 v57, v207 :: v_dual_mov_b32 v56, v206       // 0000000033C4: CA1001CF 393801CE 
	v_dual_mov_b32 v55, v205 :: v_dual_mov_b32 v54, v204       // 0000000033CC: CA1001CD 373601CC 
	v_dual_mov_b32 v51, v201 :: v_dual_mov_b32 v50, v200       // 0000000033D4: CA1001C9 333201C8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_wmma_f16_16x16x16_f16 v[240:247], v[160:167], v[137:144], v[240:247] op_sel:[0,0,1]// 000000003504: CC4260F0 1FC313A0 
	v_wmma_f16_16x16x16_f16 v[105:112], v[24:31], v[137:144], v[105:112] op_sel:[0,0,1]// 00000000350C: CC426069 1DA71318 
	s_clause 0x1                                               // 000000003514: BF850001 
	scratch_load_b128 v[137:140], off, off offset:256          // 000000003518: DC5D0100 897C0000 
	scratch_load_b128 v[141:144], off, off offset:272          // 000000003520: DC5D0110 8D7C0000 
	v_dual_mov_b32 v24, v168 :: v_dual_mov_b32 v25, v169       // 000000003528: CA1001A8 181801A9 
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 000000003530: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 000000003538: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 000000003540: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 000000003548: CA100121 A1A00120 
	v_dual_mov_b32 v30, v174 :: v_dual_mov_b32 v31, v175       // 000000003550: CA1001AE 1E1E01AF 
	v_dual_mov_b32 v26, v170 :: v_dual_mov_b32 v27, v171       // 000000003558: CA1001AA 1A1A01AB 
	v_dual_mov_b32 v28, v172 :: v_dual_mov_b32 v29, v173       // 000000003560: CA1001AC 1C1C01AD 
	v_dual_mov_b32 v175, v152 :: v_dual_mov_b32 v174, v151     // 000000003568: CA100198 AFAE0197 
	v_dual_mov_b32 v173, v150 :: v_dual_mov_b32 v172, v149     // 000000003570: CA100196 ADAC0195 
	v_dual_mov_b32 v171, v148 :: v_dual_mov_b32 v170, v147     // 000000003578: CA100194 ABAA0193 
	v_dual_mov_b32 v169, v146 :: v_dual_mov_b32 v168, v145     // 000000003580: CA100192 A9A80191 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v43, v25 :: v_dual_mov_b32 v42, v24         // 0000000035A8: CA100119 2B2A0118 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v47, v29 :: v_dual_mov_b32 v46, v28         // 0000000035F0: CA10011D 2F2E011C 
	v_dual_mov_b32 v49, v31 :: v_dual_mov_b32 v48, v30         // 0000000035F8: CA10011F 3130011E 
	v_dual_mov_b32 v45, v27 :: v_dual_mov_b32 v44, v26         // 000000003600: CA10011B 2D2C011A 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v128, off offset:384                // 000000003614: DC690180 007C8000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:540                  // 000000003630: DC69021C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:528                  // 00000000363C: DC690210 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:524                  // 000000003648: DC69020C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:544                  // 000000003654: DC690220 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:512                  // 000000003660: DC690200 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:508                  // 00000000366C: DC6901FC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:504                  // 000000003678: DC6901F8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 000000003688: DC690228 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:536                  // 000000003694: DC690218 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 0000000036A0: DC690204 007C0000 
	s_waitcnt vmcnt(1)                                         // 0000000036A8: BF8907F7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 0000000036B4: BF8903F7 
	v_dual_mov_b32 v88, v144 :: v_dual_mov_b32 v87, v143       // 0000000036B8: CA100190 5856018F 
	v_dual_mov_b32 v86, v142 :: v_dual_mov_b32 v85, v141       // 0000000036C0: CA10018E 5654018D 
	scratch_store_b32 off, v0, off offset:496                  // 0000000036C8: DC6901F0 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v84, v140 :: v_dual_mov_b32 v83, v139       // 0000000036D8: CA10018C 5452018B 
	v_dual_mov_b32 v82, v138 :: v_dual_mov_b32 v81, v137       // 0000000036E0: CA10018A 52500189 
	scratch_store_b32 off, v0, off offset:480                  // 0000000036E8: DC6901E0 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:572                  // 000000003718: DC69023C 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v144, v31 :: v_dual_mov_b32 v143, v30       // 000000003730: CA10011F 908E011E 
	v_dual_mov_b32 v142, v29 :: v_dual_mov_b32 v141, v28       // 000000003738: CA10011D 8E8C011C 
	scratch_store_b32 off, v0, off offset:468                  // 000000003740: DC6901D4 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v140, v27 :: v_dual_mov_b32 v139, v26       // 000000003750: CA10011B 8C8A011A 
	v_dual_mov_b32 v138, v25 :: v_dual_mov_b32 v137, v24       // 000000003758: CA100119 8A880118 
	scratch_store_b32 off, v0, off offset:464                  // 000000003760: DC6901D0 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 00000000378C: BF850001 
	scratch_store_b32 off, v0, off offset:460                  // 000000003790: DC6901CC 007C0000 
	scratch_store_b32 off, v29, off offset:548                 // 000000003798: DC690224 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 0000000037C4: BF850001 
	scratch_store_b32 off, v0, off offset:456                  // 0000000037C8: DC6901C8 007C0000 
	scratch_store_b32 off, v29, off offset:476                 // 0000000037D0: DC6901DC 007C1D00 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 0000000037E0: BF850001 
	scratch_load_b128 v[216:219], off, off offset:584          // 0000000037E4: DC5D0248 D87C0000 
	scratch_load_b128 v[220:223], off, off offset:600          // 0000000037EC: DC5D0258 DC7C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:452                  // 000000003804: DC6901C4 007C0000 
 <unknown>
	scratch_store_b32 off, v29, off offset:472                 // 000000003810: DC6901D8 007C1D00 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:500                  // 000000003820: DC6901F4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:492                  // 00000000382C: DC6901EC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:488                  // 000000003838: DC6901E8 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:484                  // 000000003844: DC6901E4 007C0000 
 <unknown>
	s_mov_b32 s7, 0                                            // 000000003850: BE870080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003854: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000003858: 916A077E 
	s_cbranch_vccnz 565                                        // 00000000385C: BFA40235  <_amdgpu_cs_main+0x4134>
	s_clause 0xb                                               // 000000003860: BF85000B 
	scratch_load_b128 v[0:3], off, off offset:32               // 000000003864: DC5D0020 007C0000 
	scratch_load_b128 v[4:7], off, off offset:48               // 00000000386C: DC5D0030 047C0000 
	scratch_load_b128 v[72:75], off, off offset:64             // 000000003874: DC5D0040 487C0000 
	scratch_load_b128 v[76:79], off, off offset:80             // 00000000387C: DC5D0050 4C7C0000 
	scratch_load_b128 v[248:251], off, off offset:128          // 000000003884: DC5D0080 F87C0000 
	scratch_load_b128 v[252:255], off, off offset:144          // 00000000388C: DC5D0090 FC7C0000 
	scratch_load_b128 v[160:163], off, off offset:352          // 000000003894: DC5D0160 A07C0000 
	scratch_load_b128 v[164:167], off, off offset:368          // 00000000389C: DC5D0170 A47C0000 
	scratch_load_b128 v[40:43], off, off offset:420            // 0000000038A4: DC5D01A4 287C0000 
	scratch_load_b128 v[44:47], off, off offset:436            // 0000000038AC: DC5D01B4 2C7C0000 
	scratch_load_b128 v[24:27], off, off offset:320            // 0000000038B4: DC5D0140 187C0000 
	scratch_load_b128 v[28:31], off, off offset:336            // 0000000038BC: DC5D0150 1C7C0000 
	v_dual_mov_b32 v175, v144 :: v_dual_mov_b32 v174, v143     // 0000000038C4: CA100190 AFAE018F 
	v_dual_mov_b32 v173, v142 :: v_dual_mov_b32 v170, v139     // 0000000038CC: CA10018E ADAA018B 
	v_dual_mov_b32 v171, v140 :: v_dual_mov_b32 v168, v137     // 0000000038D4: CA10018C ABA80189 
	v_mov_b32_e32 v169, v138                                   // 0000000038DC: 7F52038A 
	s_clause 0x5                                               // 0000000038E0: BF850005 
	scratch_load_b128 v[121:124], off, off offset:288          // 0000000038E4: DC5D0120 797C0000 
	scratch_load_b128 v[125:128], off, off offset:304          // 0000000038EC: DC5D0130 7D7C0000 
	scratch_load_b128 v[145:148], off, off offset:192          // 0000000038F4: DC5D00C0 917C0000 
	scratch_load_b128 v[149:152], off, off offset:208          // 0000000038FC: DC5D00D0 957C0000 
	scratch_load_b128 v[129:132], off, off offset:160          // 000000003904: DC5D00A0 817C0000 
	scratch_load_b128 v[133:136], off, off offset:176          // 00000000390C: DC5D00B0 857C0000 
	v_dual_mov_b32 v58, v232 :: v_dual_mov_b32 v59, v233       // 000000003914: CA1001E8 3A3A01E9 
	v_dual_mov_b32 v60, v234 :: v_dual_mov_b32 v61, v235       // 00000000391C: CA1001EA 3C3C01EB 
	v_dual_mov_b32 v62, v236 :: v_dual_mov_b32 v63, v237       // 000000003924: CA1001EC 3E3E01ED 
	v_dual_mov_b32 v64, v238 :: v_dual_mov_b32 v65, v239       // 00000000392C: CA1001EE 404001EF 
	s_waitcnt vmcnt(21)                                        // 000000003934: BF8957F7 
	v_dual_mov_b32 v192, v232 :: v_dual_mov_b32 v193, v233     // 000000003938: CA1001E8 C0C001E9 
	v_dual_mov_b32 v194, v234 :: v_dual_mov_b32 v195, v235     // 000000003940: CA1001EA C2C201EB 
	v_dual_mov_b32 v196, v236 :: v_dual_mov_b32 v197, v237     // 000000003948: CA1001EC C4C401ED 
	v_dual_mov_b32 v198, v238 :: v_dual_mov_b32 v199, v239     // 000000003950: CA1001EE C6C601EF 
	s_clause 0x3                                               // 000000003958: BF850003 
	scratch_load_b128 v[232:235], off, off offset:256          // 00000000395C: DC5D0100 E87C0000 
	scratch_load_b128 v[236:239], off, off offset:272          // 000000003964: DC5D0110 EC7C0000 
	scratch_load_b128 v[200:203], off, off offset:96           // 00000000396C: DC5D0060 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:112          // 000000003974: DC5D0070 CC7C0000 
	v_dual_mov_b32 v176, v184 :: v_dual_mov_b32 v177, v185     // 00000000397C: CA1001B8 B0B001B9 
	v_dual_mov_b32 v178, v186 :: v_dual_mov_b32 v179, v187     // 000000003984: CA1001BA B2B201BB 
	v_dual_mov_b32 v180, v188 :: v_dual_mov_b32 v181, v189     // 00000000398C: CA1001BC B4B401BD 
	v_dual_mov_b32 v182, v190 :: v_dual_mov_b32 v183, v191     // 000000003994: CA1001BE B6B601BF 
	s_waitcnt vmcnt(22)                                        // 00000000399C: BF895BF7 
	v_dual_mov_b32 v184, v216 :: v_dual_mov_b32 v185, v217     // 0000000039A0: CA1001D8 B8B801D9 
	v_dual_mov_b32 v190, v222 :: v_dual_mov_b32 v191, v223     // 0000000039A8: CA1001DE BEBE01DF 
	v_dual_mov_b32 v186, v218 :: v_dual_mov_b32 v187, v219     // 0000000039B0: CA1001DA BABA01DB 
	v_dual_mov_b32 v188, v220 :: v_dual_mov_b32 v189, v221     // 0000000039B8: CA1001DC BCBC01DD 
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000039C0: BF8701A3 
	v_dual_mov_b32 v223, v191 :: v_dual_mov_b32 v222, v190     // 0000000039C4: CA1001BF DFDE01BE 
	v_dual_mov_b32 v112, v191 :: v_dual_mov_b32 v111, v190     // 0000000039CC: CA1001BF 706E01BE 
	v_dual_mov_b32 v110, v189 :: v_dual_mov_b32 v109, v188     // 0000000039D4: CA1001BD 6E6C01BC 
	v_dual_mov_b32 v108, v187 :: v_dual_mov_b32 v107, v186     // 0000000039DC: CA1001BB 6C6A01BA 
	v_dual_mov_b32 v106, v185 :: v_dual_mov_b32 v105, v184     // 0000000039E4: CA1001B9 6A6801B8 
 <unknown>
 <unknown>
	v_dual_mov_b32 v221, v189 :: v_dual_mov_b32 v220, v188     // 0000000039F4: CA1001BD DDDC01BC 
	v_dual_mov_b32 v219, v187 :: v_dual_mov_b32 v218, v186     // 0000000039FC: CA1001BB DBDA01BA 
	v_dual_mov_b32 v217, v185 :: v_dual_mov_b32 v216, v184     // 000000003A04: CA1001B9 D9D801B8 
	v_dual_mov_b32 v114, v177 :: v_dual_mov_b32 v113, v176     // 000000003A0C: CA1001B1 727001B0 
	v_dual_mov_b32 v120, v183 :: v_dual_mov_b32 v119, v182     // 000000003A14: CA1001B7 787601B6 
	v_dual_mov_b32 v118, v181 :: v_dual_mov_b32 v117, v180     // 000000003A1C: CA1001B5 767401B4 
	v_dual_mov_b32 v116, v179 :: v_dual_mov_b32 v115, v178     // 000000003A24: CA1001B3 747201B2 
	v_mov_b32_e32 v172, v141                                   // 000000003A2C: 7F58038D 
	s_clause 0x1                                               // 000000003A30: BF850001 
	scratch_load_b128 v[137:140], off, off offset:388          // 000000003A34: DC5D0184 897C0000 
	scratch_load_b128 v[141:144], off, off offset:404          // 000000003A3C: DC5D0194 8D7C0000 
	s_waitcnt vmcnt(23)                                        // 000000003A44: BF895FF7 
	v_mov_b32_e32 v19, v3                                      // 000000003A48: 7E260303 
	s_waitcnt vmcnt(21)                                        // 000000003A4C: BF8957F7 
	v_dual_mov_b32 v209, v73 :: v_dual_mov_b32 v208, v72       // 000000003A50: CA100149 D1D00148 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000003A68: BF850001 
	scratch_load_b128 v[66:69], off, off                       // 000000003A6C: DC5D0000 427C0000 
	scratch_load_b128 v[70:73], off, off offset:16             // 000000003A74: DC5D0010 467C0000 
	s_waitcnt vmcnt(16)                                        // 000000003A7C: BF8943F7 
	v_wmma_f16_16x16x16_f16 v[248:255], v[160:167], v[40:47], v[248:255]// 000000003A80: CC4240F8 1FE251A0 
	s_waitcnt vmcnt(14)                                        // 000000003A88: BF893BF7 
	v_wmma_f16_16x16x16_f16 v[58:65], v[24:31], v[40:47], v[58:65]// 000000003A8C: CC42403A 1CEA5118 
	v_mov_b32_e32 v16, v0                                      // 000000003A94: 7E200300 
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1           // 000000003A98: CA100102 12100101 
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6           // 000000003AA0: CA100107 17160106 
	s_waitcnt vmcnt(12)                                        // 000000003AA8: BF8933F7 
 <unknown>
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4           // 000000003AB0: CA100105 15140104 
	s_waitcnt vmcnt(10)                                        // 000000003AB8: BF892BF7 
	v_dual_mov_b32 v104, v152 :: v_dual_mov_b32 v103, v151     // 000000003ABC: CA100198 68660197 
	scratch_store_b32 off, v0, off offset:576                  // 000000003AC4: DC690240 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v102, v150 :: v_dual_mov_b32 v101, v149     // 000000003AD4: CA100196 66640195 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:568                  // 000000003AEC: DC690238 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:532                  // 000000003B18: DC690214 007C0000 
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v191, v183 :: v_dual_mov_b32 v190, v182     // 000000003B2C: CA1001B7 BFBE01B6 
	v_dual_mov_b32 v189, v181 :: v_dual_mov_b32 v188, v180     // 000000003B34: CA1001B5 BDBC01B4 
	scratch_store_b32 off, v0, off offset:520                  // 000000003B3C: DC690208 007C0000 
	v_dual_mov_b32 v187, v179 :: v_dual_mov_b32 v186, v178     // 000000003B44: CA1001B3 BBBA01B2 
	v_dual_mov_b32 v185, v177 :: v_dual_mov_b32 v184, v176     // 000000003B4C: CA1001B1 B9B801B0 
 <unknown>
	s_waitcnt vmcnt(8)                                         // 000000003B58: BF8923F7 
	v_dual_mov_b32 v96, v136 :: v_dual_mov_b32 v95, v135       // 000000003B5C: CA100188 605E0187 
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v93, v133       // 000000003B64: CA100186 5E5C0185 
	s_waitcnt vmcnt(5)                                         // 000000003B6C: BF8917F7 
	v_dual_mov_b32 v53, v203 :: v_dual_mov_b32 v52, v202       // 000000003B70: CA1001CB 353401CA 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v88, v239 :: v_dual_mov_b32 v87, v238       // 000000003B90: CA1001EF 585601EE 
	v_dual_mov_b32 v86, v237 :: v_dual_mov_b32 v85, v236       // 000000003B98: CA1001ED 565401EC 
	v_dual_mov_b32 v84, v235 :: v_dual_mov_b32 v83, v234       // 000000003BA0: CA1001EB 545201EA 
	v_dual_mov_b32 v82, v233 :: v_dual_mov_b32 v81, v232       // 000000003BA8: CA1001E9 525001E8 
	s_waitcnt vmcnt(1)                                         // 000000003BB0: BF8907F7 
 <unknown>
 <unknown>
	v_dual_mov_b32 v92, v132 :: v_dual_mov_b32 v91, v131       // 000000003BBC: CA100184 5C5A0183 
	v_dual_mov_b32 v90, v130 :: v_dual_mov_b32 v89, v129       // 000000003BC4: CA100182 5A580181 
	v_dual_mov_b32 v100, v148 :: v_dual_mov_b32 v99, v147      // 000000003BCC: CA100194 64620193 
	v_dual_mov_b32 v98, v146 :: v_dual_mov_b32 v97, v145       // 000000003BD4: CA100192 62600191 
	v_dual_mov_b32 v215, v79 :: v_dual_mov_b32 v214, v78       // 000000003BDC: CA10014F D7D6014E 
	v_dual_mov_b32 v213, v77 :: v_dual_mov_b32 v212, v76       // 000000003BE4: CA10014D D5D4014C 
	v_dual_mov_b32 v211, v75 :: v_dual_mov_b32 v210, v74       // 000000003BEC: CA10014B D3D2014A 
	v_dual_mov_b32 v57, v207 :: v_dual_mov_b32 v56, v206       // 000000003BF4: CA1001CF 393801CE 
	v_dual_mov_b32 v55, v205 :: v_dual_mov_b32 v54, v204       // 000000003BFC: CA1001CD 373601CC 
	v_dual_mov_b32 v51, v201 :: v_dual_mov_b32 v50, v200       // 000000003C04: CA1001C9 333201C8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 000000003CB4: BF8903F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_wmma_f16_16x16x16_f16 v[248:255], v[160:167], v[137:144], v[248:255] op_sel:[0,0,1]// 000000003D3C: CC4260F8 1FE313A0 
	v_wmma_f16_16x16x16_f16 v[58:65], v[24:31], v[137:144], v[58:65] op_sel:[0,0,1]// 000000003D44: CC42603A 1CEB1318 
	s_clause 0x1                                               // 000000003D4C: BF850001 
	scratch_load_b128 v[137:140], off, off offset:224          // 000000003D50: DC5D00E0 897C0000 
	scratch_load_b128 v[141:144], off, off offset:240          // 000000003D58: DC5D00F0 8D7C0000 
	v_dual_mov_b32 v24, v168 :: v_dual_mov_b32 v25, v169       // 000000003D60: CA1001A8 181801A9 
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 000000003D68: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 000000003D70: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 000000003D78: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 000000003D80: CA100121 A1A00120 
	v_dual_mov_b32 v30, v174 :: v_dual_mov_b32 v31, v175       // 000000003D88: CA1001AE 1E1E01AF 
	v_dual_mov_b32 v26, v170 :: v_dual_mov_b32 v27, v171       // 000000003D90: CA1001AA 1A1A01AB 
	v_dual_mov_b32 v28, v172 :: v_dual_mov_b32 v29, v173       // 000000003D98: CA1001AC 1C1C01AD 
	v_dual_mov_b32 v175, v128 :: v_dual_mov_b32 v174, v127     // 000000003DA0: CA100180 AFAE017F 
 <unknown>
 <unknown>
	v_dual_mov_b32 v173, v126 :: v_dual_mov_b32 v172, v125     // 000000003DB0: CA10017E ADAC017D 
	v_dual_mov_b32 v171, v124 :: v_dual_mov_b32 v170, v123     // 000000003DB8: CA10017C ABAA017B 
	v_dual_mov_b32 v169, v122 :: v_dual_mov_b32 v168, v121     // 000000003DC0: CA10017A A9A80179 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v43, v25 :: v_dual_mov_b32 v42, v24         // 000000003E00: CA100119 2B2A0118 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v47, v29 :: v_dual_mov_b32 v46, v28         // 000000003E38: CA10011D 2F2E011C 
	v_dual_mov_b32 v49, v31 :: v_dual_mov_b32 v48, v30         // 000000003E40: CA10011F 3130011E 
	v_dual_mov_b32 v45, v27 :: v_dual_mov_b32 v44, v26         // 000000003E48: CA10011B 2D2C011A 
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v41, off offset:560                 // 000000003E5C: DC690230 007C2900 
 <unknown>
	scratch_store_b32 off, v41, off offset:564                 // 000000003E68: DC690234 007C2900 
 <unknown>
	scratch_store_b32 off, v41, off offset:556                 // 000000003E74: DC69022C 007C2900 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:540                  // 000000003E8C: DC69021C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:528                  // 000000003E98: DC690210 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:524                  // 000000003EA4: DC69020C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:544                  // 000000003EB0: DC690220 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:512                  // 000000003EBC: DC690200 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:508                  // 000000003EC8: DC6901FC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:504                  // 000000003ED4: DC6901F8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 000000003EE4: DC690228 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:536                  // 000000003EF4: DC690218 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 000000003F04: DC690204 007C0000 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 000000003F14: BF8907F7 
	v_dual_mov_b32 v243, v140 :: v_dual_mov_b32 v242, v139     // 000000003F18: CA10018C F3F2018B 
	s_waitcnt vmcnt(0)                                         // 000000003F20: BF8903F7 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:496                  // 000000003F2C: DC6901F0 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v247, v144 :: v_dual_mov_b32 v246, v143     // 000000003F3C: CA100190 F7F6018F 
	scratch_store_b32 off, v2, off offset:548                  // 000000003F44: DC690224 007C0200 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:480                  // 000000003F54: DC6901E0 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v245, v142 :: v_dual_mov_b32 v244, v141     // 000000003F64: CA10018E F5F4018D 
	v_dual_mov_b32 v241, v138 :: v_dual_mov_b32 v240, v137     // 000000003F6C: CA10018A F1F00189 
	scratch_store_b32 off, v0, off offset:572                  // 000000003F74: DC69023C 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v2, off offset:476                  // 000000003FB4: DC6901DC 007C0200 
 <unknown>
 <unknown>
	v_dual_mov_b32 v144, v31 :: v_dual_mov_b32 v143, v30       // 000000003FC4: CA10011F 908E011E 
	v_dual_mov_b32 v142, v29 :: v_dual_mov_b32 v141, v28       // 000000003FCC: CA10011D 8E8C011C 
	v_dual_mov_b32 v140, v27 :: v_dual_mov_b32 v139, v26       // 000000003FD4: CA10011B 8C8A011A 
	v_dual_mov_b32 v138, v25 :: v_dual_mov_b32 v137, v24       // 000000003FDC: CA100119 8A880118 
 <unknown>
 <unknown>
	v_dual_mov_b32 v239, v199 :: v_dual_mov_b32 v238, v198     // 000000003FEC: CA1001C7 EFEE01C6 
	v_dual_mov_b32 v237, v197 :: v_dual_mov_b32 v236, v196     // 000000003FF4: CA1001C5 EDEC01C4 
	scratch_store_b32 off, v29, off offset:468                 // 000000003FFC: DC6901D4 007C1D00 
 <unknown>
 <unknown>
	v_dual_mov_b32 v235, v195 :: v_dual_mov_b32 v234, v194     // 00000000400C: CA1001C3 EBEA01C2 
	v_dual_mov_b32 v233, v193 :: v_dual_mov_b32 v232, v192     // 000000004014: CA1001C1 E9E801C0 
	scratch_store_b32 off, v29, off offset:464                 // 00000000401C: DC6901D0 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:460                 // 00000000404C: DC6901CC 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:456                 // 00000000407C: DC6901C8 007C1D00 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 00000000408C: BF850001 
	scratch_store_b32 off, v128, off offset:384                // 000000004090: DC690180 007C8000 
	scratch_store_b32 off, v2, off offset:472                  // 000000004098: DC6901D8 007C0200 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v2, off offset:500                  // 0000000040B4: DC6901F4 007C0200 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:452                 // 0000000040F8: DC6901C4 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v2, off offset:492                  // 000000004110: DC6901EC 007C0200 
 <unknown>
	scratch_store_b32 off, v2, off offset:488                  // 00000000411C: DC6901E8 007C0200 
 <unknown>
	scratch_store_b32 off, v2, off offset:484                  // 000000004128: DC6901E4 007C0200 
 <unknown>
	s_mov_b32 s7, 0                                            // 000000004134: BE870080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004138: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 00000000413C: 916A077E 
	s_cbranch_vccnz 538                                        // 000000004140: BFA4021A  <_amdgpu_cs_main+0x49AC>
	s_clause 0x7                                               // 000000004144: BF850007 
	scratch_load_b128 v[168:171], off, off offset:352          // 000000004148: DC5D0160 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:368          // 000000004150: DC5D0170 AC7C0000 
	scratch_load_b128 v[24:27], off, off offset:420            // 000000004158: DC5D01A4 187C0000 
	scratch_load_b128 v[28:31], off, off offset:436            // 000000004160: DC5D01B4 1C7C0000 
	scratch_load_b128 v[50:53], off, off offset:96             // 000000004168: DC5D0060 327C0000 
	scratch_load_b128 v[54:57], off, off offset:112            // 000000004170: DC5D0070 367C0000 
	scratch_load_b128 v[160:163], off, off offset:320          // 000000004178: DC5D0140 A07C0000 
	scratch_load_b128 v[164:167], off, off offset:336          // 000000004180: DC5D0150 A47C0000 
	s_waitcnt vmcnt(19)                                        // 000000004188: BF894FF7 
	v_dual_mov_b32 v183, v144 :: v_dual_mov_b32 v182, v143     // 00000000418C: CA100190 B7B6018F 
	v_dual_mov_b32 v42, v137 :: v_dual_mov_b32 v43, v138       // 000000004194: CA100189 2A2A018A 
	v_dual_mov_b32 v181, v142 :: v_dual_mov_b32 v180, v141     // 00000000419C: CA10018E B5B4018D 
	v_dual_mov_b32 v179, v140 :: v_dual_mov_b32 v178, v139     // 0000000041A4: CA10018C B3B2018B 
	v_dual_mov_b32 v177, v138 :: v_dual_mov_b32 v176, v137     // 0000000041AC: CA10018A B1B00189 
	v_dual_mov_b32 v44, v139 :: v_dual_mov_b32 v45, v140       // 0000000041B4: CA10018B 2C2C018C 
	v_dual_mov_b32 v46, v141 :: v_dual_mov_b32 v47, v142       // 0000000041BC: CA10018D 2E2E018E 
	v_dual_mov_b32 v48, v143 :: v_dual_mov_b32 v49, v144       // 0000000041C4: CA10018F 30300190 
	s_clause 0xd                                               // 0000000041CC: BF85000D 
	scratch_load_b128 v[137:140], off, off offset:388          // 0000000041D0: DC5D0184 897C0000 
	scratch_load_b128 v[141:144], off, off offset:404          // 0000000041D8: DC5D0194 8D7C0000 
	scratch_load_b128 v[121:124], off, off offset:288          // 0000000041E0: DC5D0120 797C0000 
	scratch_load_b128 v[125:128], off, off offset:304          // 0000000041E8: DC5D0130 7D7C0000 
	scratch_load_b128 v[0:3], off, off offset:32               // 0000000041F0: DC5D0020 007C0000 
	scratch_load_b128 v[4:7], off, off offset:48               // 0000000041F8: DC5D0030 047C0000 
	scratch_load_b128 v[72:75], off, off offset:64             // 000000004200: DC5D0040 487C0000 
	scratch_load_b128 v[76:79], off, off offset:80             // 000000004208: DC5D0050 4C7C0000 
	scratch_load_b128 v[200:203], off, off offset:128          // 000000004210: DC5D0080 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:144          // 000000004218: DC5D0090 CC7C0000 
	scratch_load_b128 v[145:148], off, off offset:192          // 000000004220: DC5D00C0 917C0000 
	scratch_load_b128 v[149:152], off, off offset:208          // 000000004228: DC5D00D0 957C0000 
	scratch_load_b128 v[129:132], off, off offset:160          // 000000004230: DC5D00A0 817C0000 
	scratch_load_b128 v[133:136], off, off offset:176          // 000000004238: DC5D00B0 857C0000 
	s_waitcnt vmcnt(25)                                        // 000000004240: BF8967F7 
	v_dual_mov_b32 v192, v232 :: v_dual_mov_b32 v193, v233     // 000000004244: CA1001E8 C0C001E9 
	v_dual_mov_b32 v194, v234 :: v_dual_mov_b32 v195, v235     // 00000000424C: CA1001EA C2C201EB 
	v_dual_mov_b32 v196, v236 :: v_dual_mov_b32 v197, v237     // 000000004254: CA1001EC C4C401ED 
	v_dual_mov_b32 v198, v238 :: v_dual_mov_b32 v199, v239     // 00000000425C: CA1001EE C6C601EF 
	s_clause 0x1                                               // 000000004264: BF850001 
	scratch_load_b128 v[232:235], off, off offset:256          // 000000004268: DC5D0100 E87C0000 
	scratch_load_b128 v[236:239], off, off offset:272          // 000000004270: DC5D0110 EC7C0000 
	v_dual_mov_b32 v62, v196 :: v_dual_mov_b32 v63, v197       // 000000004278: CA1001C4 3E3E01C5 
	v_dual_mov_b32 v64, v198 :: v_dual_mov_b32 v65, v199       // 000000004280: CA1001C6 404001C7 
	v_dual_mov_b32 v61, v195 :: v_dual_mov_b32 v60, v194       // 000000004288: CA1001C3 3D3C01C2 
	v_dual_mov_b32 v59, v193 :: v_dual_mov_b32 v58, v192       // 000000004290: CA1001C1 3B3A01C0 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(20)                                        // 0000000042A0: BF8953F7 
	v_wmma_f16_16x16x16_f16 v[42:49], v[168:175], v[24:31], v[42:49]// 0000000042A4: CC42402A 1CAA31A8 
	s_waitcnt vmcnt(16)                                        // 0000000042AC: BF8943F7 
	v_wmma_f16_16x16x16_f16 v[50:57], v[160:167], v[24:31], v[50:57]// 0000000042B0: CC424032 1CCA31A0 
	s_clause 0x1                                               // 0000000042B8: BF850001 
	scratch_load_b128 v[24:27], off, off offset:224            // 0000000042BC: DC5D00E0 187C0000 
	scratch_load_b128 v[28:31], off, off offset:240            // 0000000042C4: DC5D00F0 1C7C0000 
	s_waitcnt vmcnt(15)                                        // 0000000042CC: BF893FF7 
 <unknown>
 <unknown>
	v_wmma_f16_16x16x16_f16 v[42:49], v[168:175], v[137:144], v[42:49] op_sel:[0,0,1]// 0000000042D8: CC42602A 1CAB13A8 
	v_wmma_f16_16x16x16_f16 v[50:57], v[160:167], v[137:144], v[50:57] op_sel:[0,0,1]// 0000000042E0: CC426032 1CCB13A0 
	v_dual_mov_b32 v137, v176 :: v_dual_mov_b32 v140, v179     // 0000000042E8: CA1001B0 898C01B3 
	s_waitcnt vmcnt(13)                                        // 0000000042F0: BF8937F7 
	v_dual_mov_b32 v143, v182 :: v_dual_mov_b32 v16, v0        // 0000000042F4: CA1001B6 8F100100 
 <unknown>
	s_waitcnt vmcnt(9)                                         // 000000004300: BF8927F7 
	v_dual_mov_b32 v249, v201 :: v_dual_mov_b32 v248, v200     // 000000004304: CA1001C9 F9F801C8 
	s_waitcnt vmcnt(5)                                         // 00000000430C: BF8917F7 
	v_dual_mov_b32 v92, v132 :: v_dual_mov_b32 v91, v131       // 000000004310: CA100184 5C5A0183 
	scratch_store_b32 off, v0, off offset:576                  // 000000004318: DC690240 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v90, v130 :: v_dual_mov_b32 v89, v129       // 000000004328: CA100182 5A580181 
	v_dual_mov_b32 v215, v79 :: v_dual_mov_b32 v214, v78       // 000000004330: CA10014F D7D6014E 
	scratch_store_b32 off, v0, off offset:568                  // 000000004338: DC690238 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v213, v77 :: v_dual_mov_b32 v212, v76       // 000000004348: CA10014D D5D4014C 
	v_dual_mov_b32 v211, v75 :: v_dual_mov_b32 v210, v74       // 000000004350: CA10014B D3D2014A 
	scratch_store_b32 off, v0, off offset:560                  // 000000004358: DC690230 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v209, v73 :: v_dual_mov_b32 v208, v72       // 000000004368: CA100149 D1D00148 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:564                  // 000000004380: DC690234 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v138, v177        // 0000000043A4: CA100103 138A01B1 
	scratch_store_b32 off, v0, off offset:556                  // 0000000043AC: DC69022C 007C0000 
 <unknown>
	v_dual_mov_b32 v139, v178 :: v_dual_mov_b32 v142, v181     // 0000000043B8: CA1001B2 8B8E01B5 
	v_dual_mov_b32 v141, v180 :: v_dual_mov_b32 v144, v183     // 0000000043C0: CA1001B4 8D9001B7 
	scratch_store_b32 off, v0, off offset:532                  // 0000000043C8: DC690214 007C0000 
 <unknown>
	s_clause 0x1                                               // 0000000043D4: BF850001 
	scratch_load_b128 v[73:76], off, off                       // 0000000043D8: DC5D0000 497C0000 
	scratch_load_b128 v[77:80], off, off offset:16             // 0000000043E0: DC5D0010 4D7C0000 
	v_dual_mov_b32 v176, v184 :: v_dual_mov_b32 v177, v185     // 0000000043E8: CA1001B8 B0B001B9 
	v_dual_mov_b32 v178, v186 :: v_dual_mov_b32 v179, v187     // 0000000043F0: CA1001BA B2B201BB 
	scratch_store_b32 off, v0, off offset:520                  // 0000000043F8: DC690208 007C0000 
	v_dual_mov_b32 v180, v188 :: v_dual_mov_b32 v181, v189     // 000000004400: CA1001BC B4B401BD 
	v_dual_mov_b32 v182, v190 :: v_dual_mov_b32 v183, v191     // 000000004408: CA1001BE B6B601BF 
	v_dual_mov_b32 v184, v216 :: v_dual_mov_b32 v185, v217     // 000000004410: CA1001D8 B8B801D9 
	v_dual_mov_b32 v190, v222 :: v_dual_mov_b32 v191, v223     // 000000004418: CA1001DE BEBE01DF 
	v_dual_mov_b32 v186, v218 :: v_dual_mov_b32 v187, v219     // 000000004420: CA1001DA BABA01DB 
	v_dual_mov_b32 v188, v220 :: v_dual_mov_b32 v189, v221     // 000000004428: CA1001DC BCBC01DD 
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004430: BF870003 
	v_dual_mov_b32 v223, v191 :: v_dual_mov_b32 v222, v190     // 000000004434: CA1001BF DFDE01BE 
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4           // 00000000443C: CA100105 15140104 
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1           // 000000004444: CA100102 12100101 
	v_dual_mov_b32 v104, v152 :: v_dual_mov_b32 v103, v151     // 00000000444C: CA100198 68660197 
	v_dual_mov_b32 v102, v150 :: v_dual_mov_b32 v101, v149     // 000000004454: CA100196 66640195 
	v_dual_mov_b32 v112, v191 :: v_dual_mov_b32 v111, v190     // 00000000445C: CA1001BF 706E01BE 
	v_dual_mov_b32 v110, v189 :: v_dual_mov_b32 v109, v188     // 000000004464: CA1001BD 6E6C01BC 
	v_dual_mov_b32 v108, v187 :: v_dual_mov_b32 v107, v186     // 00000000446C: CA1001BB 6C6A01BA 
	v_dual_mov_b32 v106, v185 :: v_dual_mov_b32 v105, v184     // 000000004474: CA1001B9 6A6801B8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 0000000044B4: BF8903F7 
 <unknown>
 <unknown>
	v_dual_mov_b32 v221, v189 :: v_dual_mov_b32 v220, v188     // 0000000044C0: CA1001BD DDDC01BC 
	v_dual_mov_b32 v219, v187 :: v_dual_mov_b32 v218, v186     // 0000000044C8: CA1001BB DBDA01BA 
	v_dual_mov_b32 v217, v185 :: v_dual_mov_b32 v216, v184     // 0000000044D0: CA1001B9 D9D801B8 
	v_dual_mov_b32 v191, v183 :: v_dual_mov_b32 v190, v182     // 0000000044D8: CA1001B7 BFBE01B6 
	v_dual_mov_b32 v114, v177 :: v_dual_mov_b32 v113, v176     // 0000000044E0: CA1001B1 727001B0 
	v_dual_mov_b32 v189, v181 :: v_dual_mov_b32 v188, v180     // 0000000044E8: CA1001B5 BDBC01B4 
	v_dual_mov_b32 v187, v179 :: v_dual_mov_b32 v186, v178     // 0000000044F0: CA1001B3 BBBA01B2 
	v_dual_mov_b32 v185, v177 :: v_dual_mov_b32 v184, v176     // 0000000044F8: CA1001B1 B9B801B0 
	v_dual_mov_b32 v175, v128 :: v_dual_mov_b32 v174, v127     // 000000004500: CA100180 AFAE017F 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v251, v203 :: v_dual_mov_b32 v250, v202     // 000000004518: CA1001CB FBFA01CA 
	v_dual_mov_b32 v88, v239 :: v_dual_mov_b32 v87, v238       // 000000004520: CA1001EF 585601EE 
	v_dual_mov_b32 v86, v237 :: v_dual_mov_b32 v85, v236       // 000000004528: CA1001ED 565401EC 
	v_dual_mov_b32 v84, v235 :: v_dual_mov_b32 v83, v234       // 000000004530: CA1001EB 545201EA 
	v_dual_mov_b32 v82, v233 :: v_dual_mov_b32 v81, v232       // 000000004538: CA1001E9 525001E8 
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v93, v133       // 000000004540: CA100186 5E5C0185 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6           // 000000004560: CA100107 17160106 
	v_dual_mov_b32 v255, v207 :: v_dual_mov_b32 v254, v206     // 000000004568: CA1001CF FFFE01CE 
	v_dual_mov_b32 v253, v205 :: v_dual_mov_b32 v252, v204     // 000000004570: CA1001CD FDFC01CC 
	v_dual_mov_b32 v96, v136 :: v_dual_mov_b32 v95, v135       // 000000004578: CA100188 605E0187 
	v_dual_mov_b32 v100, v148 :: v_dual_mov_b32 v99, v147      // 000000004580: CA100194 64620193 
	v_dual_mov_b32 v98, v146 :: v_dual_mov_b32 v97, v145       // 000000004588: CA100192 62600191 
	v_dual_mov_b32 v120, v183 :: v_dual_mov_b32 v119, v182     // 000000004590: CA1001B7 787601B6 
	v_dual_mov_b32 v118, v181 :: v_dual_mov_b32 v117, v180     // 000000004598: CA1001B5 767401B4 
	v_dual_mov_b32 v116, v179 :: v_dual_mov_b32 v115, v178     // 0000000045A0: CA1001B3 747201B2 
	v_dual_mov_b32 v173, v126 :: v_dual_mov_b32 v172, v125     // 0000000045A8: CA10017E ADAC017D 
	v_dual_mov_b32 v171, v124 :: v_dual_mov_b32 v170, v123     // 0000000045B0: CA10017C ABAA017B 
	v_dual_mov_b32 v169, v122 :: v_dual_mov_b32 v168, v121     // 0000000045B8: CA10017A A9A80179 
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 0000000045C0: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 0000000045C8: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 0000000045D0: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 0000000045D8: CA100121 A1A00120 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v128, off offset:384                // 000000004768: DC690180 007C8000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v243, v27 :: v_dual_mov_b32 v242, v26       // 000000004778: CA10011B F3F2011A 
	v_dual_mov_b32 v245, v29 :: v_dual_mov_b32 v244, v28       // 000000004780: CA10011D F5F4011C 
 <unknown>
	v_dual_mov_b32 v247, v31 :: v_dual_mov_b32 v246, v30       // 00000000478C: CA10011F F7F6011E 
	v_dual_mov_b32 v241, v25 :: v_dual_mov_b32 v240, v24       // 000000004794: CA100119 F1F00118 
	scratch_store_b32 off, v29, off offset:500                 // 00000000479C: DC6901F4 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:492                 // 0000000047C8: DC6901EC 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:488                 // 0000000047E8: DC6901E8 007C1D00 
 <unknown>
 <unknown>
	scratch_store_b32 off, v29, off offset:484                 // 0000000047F8: DC6901E4 007C1D00 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:540                  // 000000004810: DC69021C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:528                  // 00000000481C: DC690210 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:524                  // 000000004828: DC69020C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:544                  // 000000004834: DC690220 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:512                  // 000000004840: DC690200 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:508                  // 00000000484C: DC6901FC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:504                  // 000000004858: DC6901F8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 000000004868: DC690228 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:536                  // 000000004878: DC690218 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 000000004888: DC690204 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:496                  // 000000004898: DC6901F0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:480                  // 0000000048A8: DC6901E0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:572                  // 0000000048B8: DC69023C 007C0000 
 <unknown>
 <unknown>
	v_dual_mov_b32 v239, v199 :: v_dual_mov_b32 v238, v198     // 0000000048C8: CA1001C7 EFEE01C6 
	v_dual_mov_b32 v237, v197 :: v_dual_mov_b32 v236, v196     // 0000000048D0: CA1001C5 EDEC01C4 
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000048D8: BF870003 
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v235, v195 :: v_dual_mov_b32 v234, v194     // 0000000048E8: CA1001C3 EBEA01C2 
	v_dual_mov_b32 v233, v193 :: v_dual_mov_b32 v232, v192     // 0000000048F0: CA1001C1 E9E801C0 
	scratch_store_b32 off, v0, off offset:548                  // 0000000048F8: DC690224 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:476                  // 000000004924: DC6901DC 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:472                  // 000000004950: DC6901D8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:468                  // 000000004960: DC6901D4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:464                  // 000000004970: DC6901D0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:460                  // 000000004980: DC6901CC 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:456                  // 000000004990: DC6901C8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:452                  // 0000000049A0: DC6901C4 007C0000 
 <unknown>
	s_mov_b32 s7, 0                                            // 0000000049AC: BE870080 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000049B0: BF870009 
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 0000000049B4: 916A077E 
	s_cbranch_vccnz 496                                        // 0000000049B8: BFA401F0  <_amdgpu_cs_main+0x517C>
	s_clause 0x11                                              // 0000000049BC: BF850011 
	scratch_load_b128 v[16:19], off, off offset:32             // 0000000049C0: DC5D0020 107C0000 
	scratch_load_b128 v[20:23], off, off offset:48             // 0000000049C8: DC5D0030 147C0000 
	scratch_load_b128 v[168:171], off, off offset:352          // 0000000049D0: DC5D0160 A87C0000 
	scratch_load_b128 v[172:175], off, off offset:368          // 0000000049D8: DC5D0170 AC7C0000 
	scratch_load_b128 v[0:3], off, off offset:420              // 0000000049E0: DC5D01A4 007C0000 
	scratch_load_b128 v[4:7], off, off offset:436              // 0000000049E8: DC5D01B4 047C0000 
	scratch_load_b128 v[208:211], off, off offset:64           // 0000000049F0: DC5D0040 D07C0000 
	scratch_load_b128 v[212:215], off, off offset:80           // 0000000049F8: DC5D0050 D47C0000 
	scratch_load_b128 v[160:163], off, off offset:320          // 000000004A00: DC5D0140 A07C0000 
	scratch_load_b128 v[164:167], off, off offset:336          // 000000004A08: DC5D0150 A47C0000 
	scratch_load_b128 v[121:124], off, off offset:288          // 000000004A10: DC5D0120 797C0000 
	scratch_load_b128 v[125:128], off, off offset:304          // 000000004A18: DC5D0130 7D7C0000 
	scratch_load_b128 v[145:148], off, off offset:192          // 000000004A20: DC5D00C0 917C0000 
	scratch_load_b128 v[149:152], off, off offset:208          // 000000004A28: DC5D00D0 957C0000 
	scratch_load_b128 v[129:132], off, off offset:160          // 000000004A30: DC5D00A0 817C0000 
	scratch_load_b128 v[133:136], off, off offset:176          // 000000004A38: DC5D00B0 857C0000 
	scratch_load_b128 v[72:75], off, off offset:256            // 000000004A40: DC5D0100 487C0000 
	scratch_load_b128 v[76:79], off, off offset:272            // 000000004A48: DC5D0110 4C7C0000 
	v_dual_mov_b32 v24, v137 :: v_dual_mov_b32 v25, v138       // 000000004A50: CA100189 1818018A 
	v_dual_mov_b32 v26, v139 :: v_dual_mov_b32 v27, v140       // 000000004A58: CA10018B 1A1A018C 
	v_dual_mov_b32 v28, v141 :: v_dual_mov_b32 v29, v142       // 000000004A60: CA10018D 1C1C018E 
	v_dual_mov_b32 v30, v143 :: v_dual_mov_b32 v31, v144       // 000000004A68: CA10018F 1E1E0190 
	s_clause 0x3                                               // 000000004A70: BF850003 
	scratch_load_b128 v[137:140], off, off offset:128          // 000000004A74: DC5D0080 897C0000 
	scratch_load_b128 v[141:144], off, off offset:144          // 000000004A7C: DC5D0090 8D7C0000 
	scratch_load_b128 v[200:203], off, off offset:96           // 000000004A84: DC5D0060 C87C0000 
	scratch_load_b128 v[204:207], off, off offset:112          // 000000004A8C: DC5D0070 CC7C0000 
	v_dual_mov_b32 v43, v25 :: v_dual_mov_b32 v42, v24         // 000000004A94: CA100119 2B2A0118 
	v_dual_mov_b32 v45, v27 :: v_dual_mov_b32 v44, v26         // 000000004A9C: CA10011B 2D2C011A 
	v_dual_mov_b32 v49, v31 :: v_dual_mov_b32 v48, v30         // 000000004AA4: CA10011F 3130011E 
	v_dual_mov_b32 v47, v29 :: v_dual_mov_b32 v46, v28         // 000000004AAC: CA10011D 2F2E011C 
	v_dual_mov_b32 v65, v239 :: v_dual_mov_b32 v64, v238       // 000000004AB4: CA1001EF 414001EE 
	v_dual_mov_b32 v63, v237 :: v_dual_mov_b32 v62, v236       // 000000004ABC: CA1001ED 3F3E01EC 
	v_dual_mov_b32 v61, v235 :: v_dual_mov_b32 v60, v234       // 000000004AC4: CA1001EB 3D3C01EA 
	v_dual_mov_b32 v59, v233 :: v_dual_mov_b32 v58, v232       // 000000004ACC: CA1001E9 3B3A01E8 
	s_waitcnt vmcnt(22)                                        // 000000004AD4: BF895BF7 
	v_dual_mov_b32 v112, v223 :: v_dual_mov_b32 v111, v222     // 000000004AD8: CA1001DF 706E01DE 
	v_dual_mov_b32 v110, v221 :: v_dual_mov_b32 v109, v220     // 000000004AE0: CA1001DD 6E6C01DC 
	v_dual_mov_b32 v108, v219 :: v_dual_mov_b32 v107, v218     // 000000004AE8: CA1001DB 6C6A01DA 
	v_dual_mov_b32 v106, v217 :: v_dual_mov_b32 v105, v216     // 000000004AF0: CA1001D9 6A6801D8 
	v_dual_mov_b32 v120, v191 :: v_dual_mov_b32 v119, v190     // 000000004AF8: CA1001BF 787601BE 
	v_dual_mov_b32 v118, v189 :: v_dual_mov_b32 v117, v188     // 000000004B00: CA1001BD 767401BC 
	v_dual_mov_b32 v116, v187 :: v_dual_mov_b32 v115, v186     // 000000004B08: CA1001BB 747201BA 
	v_dual_mov_b32 v114, v185 :: v_dual_mov_b32 v113, v184     // 000000004B10: CA1001B9 727001B8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(16)                                        // 000000004B98: BF8943F7 
	v_wmma_f16_16x16x16_f16 v[16:23], v[168:175], v[0:7], v[16:23]// 000000004B9C: CC424010 1C4201A8 
	s_waitcnt vmcnt(12)                                        // 000000004BA4: BF8933F7 
	v_wmma_f16_16x16x16_f16 v[208:215], v[160:167], v[0:7], v[208:215]// 000000004BA8: CC4240D0 1F4201A0 
	s_clause 0x1                                               // 000000004BB0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:388              // 000000004BB4: DC5D0184 007C0000 
	scratch_load_b128 v[4:7], off, off offset:404              // 000000004BBC: DC5D0194 047C0000 
	s_waitcnt vmcnt(13)                                        // 000000004BC4: BF8937F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(12)                                        // 000000004BE8: BF8933F7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(7)                                         // 000000004BFC: BF891FF7 
	v_dual_mov_b32 v82, v73 :: v_dual_mov_b32 v81, v72         // 000000004C00: CA100149 52500148 
	v_dual_mov_b32 v92, v132 :: v_dual_mov_b32 v91, v131       // 000000004C08: CA100184 5C5A0183 
	s_waitcnt vmcnt(5)                                         // 000000004C10: BF8917F7 
	v_dual_mov_b32 v249, v138 :: v_dual_mov_b32 v248, v137     // 000000004C14: CA10018A F9F80189 
	v_dual_mov_b32 v90, v130 :: v_dual_mov_b32 v89, v129       // 000000004C1C: CA100182 5A580181 
	v_dual_mov_b32 v96, v136 :: v_dual_mov_b32 v95, v135       // 000000004C24: CA100188 605E0187 
	s_waitcnt vmcnt(4)                                         // 000000004C2C: BF8913F7 
	v_dual_mov_b32 v255, v144 :: v_dual_mov_b32 v254, v143     // 000000004C30: CA100190 FFFE018F 
	v_dual_mov_b32 v253, v142 :: v_dual_mov_b32 v252, v141     // 000000004C38: CA10018E FDFC018D 
	v_dual_mov_b32 v251, v140 :: v_dual_mov_b32 v250, v139     // 000000004C40: CA10018C FBFA018B 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v88, v79 :: v_dual_mov_b32 v87, v78         // 000000004C60: CA10014F 5856014E 
	v_dual_mov_b32 v86, v77 :: v_dual_mov_b32 v85, v76         // 000000004C68: CA10014D 5654014C 
	v_dual_mov_b32 v84, v75 :: v_dual_mov_b32 v83, v74         // 000000004C70: CA10014B 5452014A 
	v_dual_mov_b32 v94, v134 :: v_dual_mov_b32 v93, v133       // 000000004C78: CA100186 5E5C0185 
	v_dual_mov_b32 v104, v152 :: v_dual_mov_b32 v103, v151     // 000000004C80: CA100198 68660197 
	v_dual_mov_b32 v102, v150 :: v_dual_mov_b32 v101, v149     // 000000004C88: CA100196 66640195 
	v_dual_mov_b32 v100, v148 :: v_dual_mov_b32 v99, v147      // 000000004C90: CA100194 64620193 
	v_dual_mov_b32 v98, v146 :: v_dual_mov_b32 v97, v145       // 000000004C98: CA100192 62600191 
	s_waitcnt vmcnt(2)                                         // 000000004CA0: BF890BF7 
	v_dual_mov_b32 v57, v207 :: v_dual_mov_b32 v56, v206       // 000000004CA4: CA1001CF 393801CE 
	v_dual_mov_b32 v55, v205 :: v_dual_mov_b32 v54, v204       // 000000004CAC: CA1001CD 373601CC 
	v_dual_mov_b32 v53, v203 :: v_dual_mov_b32 v52, v202       // 000000004CB4: CA1001CB 353401CA 
	v_dual_mov_b32 v51, v201 :: v_dual_mov_b32 v50, v200       // 000000004CBC: CA1001C9 333201C8 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 000000004DAC: BF8903F7 
	v_wmma_f16_16x16x16_f16 v[16:23], v[168:175], v[0:7], v[16:23] op_sel:[0,0,1]// 000000004DB0: CC426010 1C4201A8 
	v_wmma_f16_16x16x16_f16 v[208:215], v[160:167], v[0:7], v[208:215] op_sel:[0,0,1]// 000000004DB8: CC4260D0 1F4201A0 
	s_clause 0x1                                               // 000000004DC0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:224              // 000000004DC4: DC5D00E0 007C0000 
	scratch_load_b128 v[4:7], off, off offset:240              // 000000004DCC: DC5D00F0 047C0000 
	v_dual_mov_b32 v175, v128 :: v_dual_mov_b32 v174, v127     // 000000004DD4: CA100180 AFAE017F 
	v_dual_mov_b32 v173, v126 :: v_dual_mov_b32 v172, v125     // 000000004DDC: CA10017E ADAC017D 
	v_dual_mov_b32 v171, v124 :: v_dual_mov_b32 v170, v123     // 000000004DE4: CA10017C ABAA017B 
	v_dual_mov_b32 v169, v122 :: v_dual_mov_b32 v168, v121     // 000000004DEC: CA10017A A9A80179 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v167, v39 :: v_dual_mov_b32 v166, v38       // 000000004E04: CA100127 A7A60126 
	v_dual_mov_b32 v165, v37 :: v_dual_mov_b32 v164, v36       // 000000004E0C: CA100125 A5A40124 
	v_dual_mov_b32 v163, v35 :: v_dual_mov_b32 v162, v34       // 000000004E14: CA100123 A3A20122 
	v_dual_mov_b32 v161, v33 :: v_dual_mov_b32 v160, v32       // 000000004E1C: CA100121 A1A00120 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v72, off offset:532                 // 000000004E54: DC690214 007C4800 
 <unknown>
 <unknown>
	scratch_store_b32 off, v136, off offset:520                // 000000004E64: DC690208 007C8800 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 000000004E74: BF8907F7 
	v_dual_mov_b32 v241, v1 :: v_dual_mov_b32 v240, v0         // 000000004E78: CA100101 F1F00100 
 <unknown>
	s_waitcnt vmcnt(0)                                         // 000000004E84: BF8903F7 
	v_dual_mov_b32 v247, v7 :: v_dual_mov_b32 v246, v6         // 000000004E88: CA100107 F7F60106 
	v_dual_mov_b32 v245, v5 :: v_dual_mov_b32 v244, v4         // 000000004E90: CA100105 F5F40104 
	scratch_store_b32 off, v0, off offset:576                  // 000000004E98: DC690240 007C0000 
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000004EA8: BF850001 
	scratch_load_b128 v[121:124], off, off                     // 000000004EAC: DC5D0000 797C0000 
	scratch_load_b128 v[125:128], off, off offset:16           // 000000004EB4: DC5D0010 7D7C0000 
	s_waitcnt vmcnt(0)                                         // 000000004EBC: BF8903F7 
 <unknown>
 <unknown>
	v_dual_mov_b32 v243, v3 :: v_dual_mov_b32 v242, v2         // 000000004EC8: CA100103 F3F20102 
	scratch_store_b32 off, v0, off offset:568                  // 000000004ED0: DC690238 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:560                  // 000000004F00: DC690230 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:564                  // 000000004F30: DC690234 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:556                  // 000000004F5C: DC69022C 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:540                  // 000000004F78: DC69021C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:528                  // 000000004F84: DC690210 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:524                  // 000000004F90: DC69020C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:544                  // 000000004F9C: DC690220 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:512                  // 000000004FA8: DC690200 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:508                  // 000000004FB4: DC6901FC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:504                  // 000000004FC0: DC6901F8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 000000004FD0: DC690228 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:536                  // 000000004FE0: DC690218 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 000000004FF0: DC690204 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:496                  // 000000004FFC: DC6901F0 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:480                  // 000000005008: DC6901E0 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:572                  // 000000005014: DC69023C 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:548                  // 000000005020: DC690224 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:476                  // 00000000502C: DC6901DC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:472                  // 000000005038: DC6901D8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:468                  // 000000005048: DC6901D4 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:464                  // 000000005058: DC6901D0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:460                  // 000000005068: DC6901CC 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:456                  // 000000005078: DC6901C8 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:452                  // 000000005088: DC6901C4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:500                  // 000000005094: DC6901F4 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:492                  // 0000000050A0: DC6901EC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:488                  // 0000000050AC: DC6901E8 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:484                  // 0000000050B8: DC6901E4 007C0000 
	v_dual_mov_b32 v0, v24 :: v_dual_mov_b32 v1, v25           // 0000000050C0: CA100118 00000119 
	v_dual_mov_b32 v6, v30 :: v_dual_mov_b32 v7, v31           // 0000000050C8: CA10011E 0606011F 
	v_dual_mov_b32 v2, v26 :: v_dual_mov_b32 v3, v27           // 0000000050D0: CA10011A 0202011B 
	v_dual_mov_b32 v4, v28 :: v_dual_mov_b32 v5, v29           // 0000000050D8: CA10011C 0404011D 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v144, v7 :: v_dual_mov_b32 v143, v6         // 0000000050F8: CA100107 908E0106 
 <unknown>
 <unknown>
	scratch_store_b32 off, v128, off offset:384                // 000000005108: DC690180 007C8000 
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v142, v5 :: v_dual_mov_b32 v141, v4         // 00000000511C: CA100105 8E8C0104 
	v_dual_mov_b32 v140, v3 :: v_dual_mov_b32 v139, v2         // 000000005124: CA100103 8C8A0102 
	v_dual_mov_b32 v138, v1 :: v_dual_mov_b32 v137, v0         // 00000000512C: CA100101 8A880100 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_cbranch_execnz 60582                                     // 00000000517C: BFA6ECA6  <_amdgpu_cs_main+0x418>
	s_branch 0                                                 // 000000005180: BFA00000  <_amdgpu_cs_main+0x5184>
	s_clause 0x9                                               // 000000005184: BF850009 
	scratch_load_b128 v[24:27], off, off offset:352            // 000000005188: DC5D0160 187C0000 
	scratch_load_b128 v[28:31], off, off offset:368            // 000000005190: DC5D0170 1C7C0000 
	scratch_load_b128 v[129:132], off, off offset:420          // 000000005198: DC5D01A4 817C0000 
	scratch_load_b128 v[133:136], off, off offset:436          // 0000000051A0: DC5D01B4 857C0000 
	scratch_load_b128 v[16:19], off, off offset:320            // 0000000051A8: DC5D0140 107C0000 
	scratch_load_b128 v[20:23], off, off offset:336            // 0000000051B0: DC5D0150 147C0000 
	scratch_load_b128 v[145:148], off, off                     // 0000000051B8: DC5D0000 917C0000 
	scratch_load_b128 v[149:152], off, off offset:16           // 0000000051C0: DC5D0010 957C0000 
	scratch_load_b128 v[0:3], off, off offset:388              // 0000000051C8: DC5D0184 007C0000 
	scratch_load_b128 v[4:7], off, off offset:404              // 0000000051D0: DC5D0194 047C0000 
	v_dual_mov_b32 v160, v32 :: v_dual_mov_b32 v161, v33       // 0000000051D8: CA100120 A0A00121 
	v_dual_mov_b32 v162, v34 :: v_dual_mov_b32 v163, v35       // 0000000051E0: CA100122 A2A20123 
	v_dual_mov_b32 v164, v36 :: v_dual_mov_b32 v165, v37       // 0000000051E8: CA100124 A4A40125 
	v_dual_mov_b32 v166, v38 :: v_dual_mov_b32 v167, v39       // 0000000051F0: CA100126 A6A60127 
	s_waitcnt vmcnt(21)                                        // 0000000051F8: BF8957F7 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(11)                                        // 000000005204: BF892FF7 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v113, v184 :: v_dual_mov_b32 v114, v185     // 000000005224: CA1001B8 717201B9 
	v_dual_mov_b32 v115, v186 :: v_dual_mov_b32 v116, v187     // 00000000522C: CA1001BA 737401BB 
	v_dual_mov_b32 v117, v188 :: v_dual_mov_b32 v118, v189     // 000000005234: CA1001BC 757601BD 
	v_dual_mov_b32 v119, v190 :: v_dual_mov_b32 v120, v191     // 00000000523C: CA1001BE 777801BF 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v105, v216 :: v_dual_mov_b32 v106, v217     // 000000005284: CA1001D8 696A01D9 
	v_dual_mov_b32 v107, v218 :: v_dual_mov_b32 v108, v219     // 00000000528C: CA1001DA 6B6C01DB 
	s_waitcnt vmcnt(10)                                        // 000000005294: BF892BF7 
	v_dual_mov_b32 v109, v220 :: v_dual_mov_b32 v110, v221     // 000000005298: CA1001DC 6D6E01DD 
	v_dual_mov_b32 v111, v222 :: v_dual_mov_b32 v112, v223     // 0000000052A0: CA1001DE 6F7001DF 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	v_dual_mov_b32 v58, v232 :: v_dual_mov_b32 v59, v233       // 0000000052D0: CA1001E8 3A3A01E9 
	v_dual_mov_b32 v60, v234 :: v_dual_mov_b32 v61, v235       // 0000000052D8: CA1001EA 3C3C01EB 
	v_dual_mov_b32 v62, v236 :: v_dual_mov_b32 v63, v237       // 0000000052E0: CA1001EC 3E3E01ED 
	v_dual_mov_b32 v64, v238 :: v_dual_mov_b32 v65, v239       // 0000000052E8: CA1001EE 404001EF 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(6)                                         // 0000000052F8: BF891BF7 
	v_wmma_f16_16x16x16_f16 v[8:15], v[24:31], v[129:136], v[8:15]// 0000000052FC: CC424008 1C230318 
	s_waitcnt vmcnt(2)                                         // 000000005304: BF890BF7 
	v_wmma_f16_16x16x16_f16 v[145:152], v[16:23], v[129:136], v[145:152]// 000000005308: CC424091 1E470310 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(0)                                         // 000000005328: BF8903F7 
	v_wmma_f16_16x16x16_f16 v[8:15], v[24:31], v[0:7], v[8:15] op_sel:[0,0,1]// 00000000532C: CC426008 1C220118 
	v_wmma_f16_16x16x16_f16 v[145:152], v[16:23], v[0:7], v[145:152] op_sel:[0,0,1]// 000000005334: CC426091 1E460110 
	s_clause 0x1                                               // 00000000533C: BF850001 
	scratch_load_b128 v[0:3], off, off offset:288              // 000000005340: DC5D0120 007C0000 
	scratch_load_b128 v[4:7], off, off offset:304              // 000000005348: DC5D0130 047C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:384                 // 00000000536C: DC690180 007C2000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:540                 // 00000000537C: DC69021C 007C2000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:528                 // 00000000538C: DC690210 007C2000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:524                 // 00000000539C: DC69020C 007C2000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:544                 // 0000000053AC: DC690220 007C2000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:512                 // 0000000053BC: DC690200 007C2000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v32, off offset:508                 // 0000000053CC: DC6901FC 007C2000 
 <unknown>
 <unknown>
	s_clause 0x2                                               // 0000000053DC: BF850002 
	scratch_store_b32 off, v32, off offset:504                 // 0000000053E0: DC6901F8 007C2000 
	scratch_store_b128 off, v[145:148], off                    // 0000000053E8: DC750000 007C9100 
	scratch_store_b128 off, v[149:152], off offset:16          // 0000000053F0: DC750010 007C9500 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 000000005430: BF8907F7 
	v_dual_mov_b32 v168, v0 :: v_dual_mov_b32 v169, v1         // 000000005434: CA100100 A8A80101 
	s_waitcnt vmcnt(0)                                         // 00000000543C: BF8903F7 
 <unknown>
	v_dual_mov_b32 v170, v2 :: v_dual_mov_b32 v171, v3         // 000000005444: CA100102 AAAA0103 
	v_dual_mov_b32 v172, v4 :: v_dual_mov_b32 v173, v5         // 00000000544C: CA100104 ACAC0105 
	scratch_store_b32 off, v0, off offset:576                  // 000000005454: DC690240 007C0000 
 <unknown>
	v_dual_mov_b32 v174, v6 :: v_dual_mov_b32 v175, v7         // 000000005460: CA100106 AEAE0107 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:568                  // 000000005478: DC690238 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:560                  // 0000000054A8: DC690230 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:564                  // 0000000054B8: DC690234 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:556                  // 0000000054C8: DC69022C 007C0000 
	s_clause 0x1                                               // 0000000054D0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:96               // 0000000054D4: DC5D0060 007C0000 
	scratch_load_b128 v[4:7], off, off offset:112              // 0000000054DC: DC5D0070 047C0000 
 <unknown>
 <unknown>
	s_waitcnt vmcnt(1)                                         // 0000000054EC: BF8907F7 
	v_dual_mov_b32 v50, v0 :: v_dual_mov_b32 v51, v1           // 0000000054F0: CA100100 32320101 
	v_dual_mov_b32 v52, v2 :: v_dual_mov_b32 v53, v3           // 0000000054F8: CA100102 34340103 
	s_waitcnt vmcnt(0)                                         // 000000005500: BF8903F7 
	v_dual_mov_b32 v54, v4 :: v_dual_mov_b32 v55, v5           // 000000005504: CA100104 36360105 
	v_dual_mov_b32 v56, v6 :: v_dual_mov_b32 v57, v7           // 00000000550C: CA100106 38380107 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 000000005544: BF850001 
	scratch_load_b128 v[0:3], off, off offset:64               // 000000005548: DC5D0040 007C0000 
	scratch_load_b128 v[4:7], off, off offset:80               // 000000005550: DC5D0050 047C0000 
	s_waitcnt vmcnt(1)                                         // 000000005558: BF8907F7 
	v_dual_mov_b32 v208, v0 :: v_dual_mov_b32 v209, v1         // 00000000555C: CA100100 D0D00101 
 <unknown>
	v_dual_mov_b32 v210, v2 :: v_dual_mov_b32 v211, v3         // 000000005568: CA100102 D2D20103 
	s_waitcnt vmcnt(0)                                         // 000000005570: BF8903F7 
	v_dual_mov_b32 v212, v4 :: v_dual_mov_b32 v213, v5         // 000000005574: CA100104 D4D40105 
	scratch_store_b32 off, v0, off offset:532                  // 00000000557C: DC690214 007C0000 
 <unknown>
	v_dual_mov_b32 v214, v6 :: v_dual_mov_b32 v215, v7         // 000000005588: CA100106 D6D60107 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:520                  // 0000000055BC: DC690208 007C0000 
	s_clause 0x1                                               // 0000000055C4: BF850001 
	scratch_load_b128 v[0:3], off, off offset:192              // 0000000055C8: DC5D00C0 007C0000 
	scratch_load_b128 v[4:7], off, off offset:208              // 0000000055D0: DC5D00D0 047C0000 
	s_waitcnt vmcnt(1)                                         // 0000000055D8: BF8907F7 
	v_dual_mov_b32 v97, v0 :: v_dual_mov_b32 v98, v1           // 0000000055DC: CA100100 61620101 
 <unknown>
	v_dual_mov_b32 v99, v2 :: v_dual_mov_b32 v100, v3          // 0000000055E8: CA100102 63640103 
	s_waitcnt vmcnt(0)                                         // 0000000055F0: BF8903F7 
	v_dual_mov_b32 v101, v4 :: v_dual_mov_b32 v102, v5         // 0000000055F4: CA100104 65660105 
	v_dual_mov_b32 v103, v6 :: v_dual_mov_b32 v104, v7         // 0000000055FC: CA100106 67680107 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:552                  // 00000000560C: DC690228 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 00000000563C: BF850001 
	scratch_load_b128 v[0:3], off, off offset:160              // 000000005640: DC5D00A0 007C0000 
	scratch_load_b128 v[4:7], off, off offset:176              // 000000005648: DC5D00B0 047C0000 
	s_waitcnt vmcnt(1)                                         // 000000005650: BF8907F7 
	v_dual_mov_b32 v89, v0 :: v_dual_mov_b32 v90, v1           // 000000005654: CA100100 595A0101 
 <unknown>
	v_dual_mov_b32 v91, v2 :: v_dual_mov_b32 v92, v3           // 000000005660: CA100102 5B5C0103 
	s_waitcnt vmcnt(0)                                         // 000000005668: BF8903F7 
	v_dual_mov_b32 v93, v4 :: v_dual_mov_b32 v94, v5           // 00000000566C: CA100104 5D5E0105 
	scratch_store_b32 off, v0, off offset:536                  // 000000005674: DC690218 007C0000 
 <unknown>
	v_dual_mov_b32 v95, v6 :: v_dual_mov_b32 v96, v7           // 000000005680: CA100106 5F600107 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:516                  // 000000005690: DC690204 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_clause 0x1                                               // 0000000056B0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:256              // 0000000056B4: DC5D0100 007C0000 
	scratch_load_b128 v[4:7], off, off offset:272              // 0000000056BC: DC5D0110 047C0000 
	s_waitcnt vmcnt(1)                                         // 0000000056C4: BF8907F7 
	v_dual_mov_b32 v81, v0 :: v_dual_mov_b32 v82, v1           // 0000000056C8: CA100100 51520101 
 <unknown>
	v_dual_mov_b32 v83, v2 :: v_dual_mov_b32 v84, v3           // 0000000056D4: CA100102 53540103 
	s_waitcnt vmcnt(0)                                         // 0000000056DC: BF8903F7 
	v_dual_mov_b32 v85, v4 :: v_dual_mov_b32 v86, v5           // 0000000056E0: CA100104 55560105 
	scratch_store_b32 off, v0, off offset:496                  // 0000000056E8: DC6901F0 007C0000 
 <unknown>
	v_dual_mov_b32 v87, v6 :: v_dual_mov_b32 v88, v7           // 0000000056F4: CA100106 57580107 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:480                  // 00000000570C: DC6901E0 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:572                  // 00000000572C: DC69023C 007C0000 
	s_clause 0x1                                               // 000000005734: BF850001 
	scratch_load_b128 v[0:3], off, off offset:224              // 000000005738: DC5D00E0 007C0000 
	scratch_load_b128 v[4:7], off, off offset:240              // 000000005740: DC5D00F0 047C0000 
	s_waitcnt vmcnt(1)                                         // 000000005748: BF8907F7 
	v_dual_mov_b32 v240, v0 :: v_dual_mov_b32 v241, v1         // 00000000574C: CA100100 F0F00101 
	s_waitcnt vmcnt(0)                                         // 000000005754: BF8903F7 
 <unknown>
	v_dual_mov_b32 v242, v2 :: v_dual_mov_b32 v243, v3         // 00000000575C: CA100102 F2F20103 
	v_dual_mov_b32 v244, v4 :: v_dual_mov_b32 v245, v5         // 000000005764: CA100104 F4F40105 
	scratch_store_b32 off, v0, off offset:548                  // 00000000576C: DC690224 007C0000 
 <unknown>
	v_dual_mov_b32 v246, v6 :: v_dual_mov_b32 v247, v7         // 000000005778: CA100106 F6F60107 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:476                  // 000000005790: DC6901DC 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:472                  // 0000000057A8: DC6901D8 007C0000 
	s_clause 0x1                                               // 0000000057B0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:128              // 0000000057B4: DC5D0080 007C0000 
	scratch_load_b128 v[4:7], off, off offset:144              // 0000000057BC: DC5D0090 047C0000 
	s_waitcnt vmcnt(1)                                         // 0000000057C4: BF8907F7 
	v_dual_mov_b32 v248, v0 :: v_dual_mov_b32 v249, v1         // 0000000057C8: CA100100 F8F80101 
 <unknown>
	v_dual_mov_b32 v250, v2 :: v_dual_mov_b32 v251, v3         // 0000000057D4: CA100102 FAFA0103 
	s_waitcnt vmcnt(0)                                         // 0000000057DC: BF8903F7 
	v_dual_mov_b32 v252, v4 :: v_dual_mov_b32 v253, v5         // 0000000057E0: CA100104 FCFC0105 
	scratch_store_b32 off, v0, off offset:468                  // 0000000057E8: DC6901D4 007C0000 
 <unknown>
	v_dual_mov_b32 v254, v6 :: v_dual_mov_b32 v255, v7         // 0000000057F4: CA100106 FEFE0107 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:464                  // 000000005804: DC6901D0 007C0000 
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:460                  // 000000005814: DC6901CC 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:456                  // 000000005820: DC6901C8 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:452                  // 00000000582C: DC6901C4 007C0000 
	v_dual_mov_b32 v0, v137 :: v_dual_mov_b32 v1, v138         // 000000005834: CA100189 0000018A 
	v_dual_mov_b32 v2, v139 :: v_dual_mov_b32 v3, v140         // 00000000583C: CA10018B 0202018C 
	v_dual_mov_b32 v4, v141 :: v_dual_mov_b32 v5, v142         // 000000005844: CA10018D 0404018E 
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000584C: BF870003 
	v_dual_mov_b32 v42, v0 :: v_dual_mov_b32 v43, v1           // 000000005850: CA100100 2A2A0101 
 <unknown>
	v_dual_mov_b32 v6, v143 :: v_dual_mov_b32 v7, v144         // 00000000585C: CA10018F 06060190 
	v_dual_mov_b32 v44, v2 :: v_dual_mov_b32 v45, v3           // 000000005864: CA100102 2C2C0103 
	scratch_store_b32 off, v0, off offset:500                  // 00000000586C: DC6901F4 007C0000 
 <unknown>
	v_dual_mov_b32 v46, v4 :: v_dual_mov_b32 v47, v5           // 000000005878: CA100104 2E2E0105 
	v_dual_mov_b32 v48, v6 :: v_dual_mov_b32 v49, v7           // 000000005880: CA100106 30300107 
	scratch_store_b32 off, v0, off offset:492                  // 000000005888: DC6901EC 007C0000 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	scratch_store_b32 off, v0, off offset:488                  // 0000000058AC: DC6901E8 007C0000 
 <unknown>
	scratch_store_b32 off, v0, off offset:484                  // 0000000058B8: DC6901E4 007C0000 
	s_clause 0x1                                               // 0000000058C0: BF850001 
	scratch_load_b128 v[0:3], off, off offset:32               // 0000000058C4: DC5D0020 007C0000 
	scratch_load_b128 v[4:7], off, off offset:48               // 0000000058CC: DC5D0030 047C0000 
	s_waitcnt vmcnt(1)                                         // 0000000058D4: BF8907F7 
	v_dual_mov_b32 v16, v0 :: v_dual_mov_b32 v17, v1           // 0000000058D8: CA100100 10100101 
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v19, v3           // 0000000058E0: CA100102 12120103 
	s_waitcnt vmcnt(0)                                         // 0000000058E8: BF8903F7 
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v21, v5           // 0000000058EC: CA100104 14140105 
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v23, v7           // 0000000058F4: CA100106 16160107 
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
 <unknown>
	s_branch 60092                                             // 000000005924: BFA0EABC  <_amdgpu_cs_main+0x418>
	v_lshlrev_b32_e32 v41, 1, v129                             // 000000005928: 30530281 
	v_ashrrev_i32_e32 v32, 31, v129                            // 00000000592C: 3441029F 
	v_and_b32_e32 v33, 0x7ffffff0, v129                        // 000000005930: 364302FF 7FFFFFF0 
	v_cmp_gt_u32_e32 vcc_lo, 0x100, v129                       // 000000005938: 7C9902FF 00000100 
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005940: BF870093 
	v_lshrrev_b32_e32 v32, 28, v32                             // 000000005944: 3240409C 
	v_add_nc_u32_e32 v32, v129, v32                            // 000000005948: 4A404181 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000594C: BF870091 
	v_and_b32_e32 v32, 0x7ffffff0, v32                         // 000000005950: 364040FF 7FFFFFF0 
	v_sub_nc_u32_e32 v32, v129, v32                            // 000000005958: 4C404181 
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000595C: BF870091 
	v_lshlrev_b32_e32 v32, 1, v32                              // 000000005960: 30404081 
	v_lshl_add_u32 v33, v33, 1, v32                            // 000000005964: D6460021 04810321 
	ds_store_b16 v33, v8 offset:2048                           // 00000000596C: D87C0800 00000821 
	ds_store_b16 v33, v9 offset:2112                           // 000000005974: D87C0840 00000921 
	ds_store_b16 v33, v10 offset:2176                          // 00000000597C: D87C0880 00000A21 
	ds_store_b16 v33, v11 offset:2240                          // 000000005984: D87C08C0 00000B21 
	ds_store_b16 v33, v12 offset:2304                          // 00000000598C: D87C0900 00000C21 
	ds_store_b16 v33, v13 offset:2368                          // 000000005994: D87C0940 00000D21 
	ds_store_b16 v33, v14 offset:2432                          // 00000000599C: D87C0980 00000E21 
	ds_store_b16 v33, v15 offset:2496                          // 0000000059A4: D87C09C0 00000F21 
	v_add_nc_u32_e32 v32, 0x800, v33                           // 0000000059AC: 4A4042FF 00000800 
	s_waitcnt lgkmcnt(0)                                       // 0000000059B4: BF89FC07 
	s_waitcnt_vscnt null, 0x0                                  // 0000000059B8: BC7C0000 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000059BC: BE83206A 
	s_cbranch_execz 77                                         // 0000000059C0: BFA5004D  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v41 offset:2048                       // 0000000059C4: DA980800 23000029 
	v_add_nc_u32_e32 v33, 0x800, v41                           // 0000000059CC: 4A4252FF 00000800 
	v_add_nc_u32_e32 v34, 0xa00, v41                           // 0000000059D4: 4A4452FF 00000A00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000059DC: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000059E8: BF89FC07 
	ds_store_b16 v41, v35 offset:2560                          // 0000000059EC: D87C0A00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000059F4: 8B7E007E 
	s_cbranch_execz 63                                         // 0000000059F8: BFA5003F  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v33 offset:64                         // 0000000059FC: DA980040 23000021 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000005A04: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005A10: BF89FC07 
	ds_store_b16 v34, v35 offset:64                            // 000000005A14: D87C0040 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005A1C: 8B7E007E 
	s_cbranch_execz 53                                         // 000000005A20: BFA50035  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v33 offset:128                        // 000000005A24: DA980080 23000021 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000005A2C: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005A38: BF89FC07 
	ds_store_b16 v34, v35 offset:128                           // 000000005A3C: D87C0080 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005A44: 8B7E007E 
	s_cbranch_execz 43                                         // 000000005A48: BFA5002B  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000005A4C: DA9800C0 23000021 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000005A54: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000005A60: BF89FC07 
	ds_store_b16 v34, v35 offset:192                           // 000000005A64: D87C00C0 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005A6C: 8B7E007E 
	s_cbranch_execz 33                                         // 000000005A70: BFA50021  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000005A74: DA980100 23000021 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000005A7C: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000005A88: BF89FC07 
	ds_store_b16 v34, v35 offset:256                           // 000000005A8C: D87C0100 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005A94: 8B7E007E 
	s_cbranch_execz 23                                         // 000000005A98: BFA50017  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000005A9C: DA980140 23000021 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000005AA4: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005AAC: BF89FC07 
	ds_store_b16 v34, v35 offset:320                           // 000000005AB0: D87C0140 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005AB8: 8B7E007E 
	s_cbranch_execz 14                                         // 000000005ABC: BFA5000E  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v35, v33 offset:384                        // 000000005AC0: DA980180 23000021 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000005AC8: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005AD0: BF89FC07 
	ds_store_b16 v34, v35 offset:384                           // 000000005AD4: D87C0180 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005ADC: 8B7E007E 
	s_cbranch_execz 5                                          // 000000005AE0: BFA50005  <_amdgpu_cs_main+0x5AF8>
	ds_load_u16_d16 v33, v33 offset:448                        // 000000005AE4: DA9801C0 21000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005AEC: BF89FC07 
	ds_store_b16 v34, v33 offset:448                           // 000000005AF0: D87C01C0 00002122 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005AF8: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000005AFC: BF89FC07 
	ds_store_b16 v32, v16                                      // 000000005B00: D87C0000 00001020 
	ds_store_b16 v32, v17 offset:64                            // 000000005B08: D87C0040 00001120 
	ds_store_b16 v32, v18 offset:128                           // 000000005B10: D87C0080 00001220 
	ds_store_b16 v32, v19 offset:192                           // 000000005B18: D87C00C0 00001320 
	ds_store_b16 v32, v20 offset:256                           // 000000005B20: D87C0100 00001420 
	ds_store_b16 v32, v21 offset:320                           // 000000005B28: D87C0140 00001520 
	ds_store_b16 v32, v22 offset:384                           // 000000005B30: D87C0180 00001620 
	ds_store_b16 v32, v23 offset:448                           // 000000005B38: D87C01C0 00001720 
	s_waitcnt lgkmcnt(0)                                       // 000000005B40: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000005B44: BE83206A 
	s_cbranch_execz 108                                        // 000000005B48: BFA5006C  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v41 offset:2560                       // 000000005B4C: DA980A00 23000029 
	s_waitcnt lgkmcnt(0)                                       // 000000005B54: BF89FC07 
	ds_load_u16_d16_hi v35, v41 offset:2048                    // 000000005B58: DA9C0800 23000029 
	v_add_nc_u32_e32 v33, 0xa00, v41                           // 000000005B60: 4A4252FF 00000A00 
	v_add_nc_u32_e32 v34, 0x800, v41                           // 000000005B68: 4A4452FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000005B70: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000005B7C: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005B80: 644647A3 
	ds_store_b16 v41, v35 offset:2560                          // 000000005B84: D87C0A00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005B8C: 8B7E007E 
	s_cbranch_execz 90                                         // 000000005B90: BFA5005A  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:64                         // 000000005B94: DA980040 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005B9C: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:64                      // 000000005BA0: DA9C0040 23000022 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000005BA8: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005BB4: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005BB8: 644647A3 
	ds_store_b16 v33, v35 offset:64                            // 000000005BBC: D87C0040 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005BC4: 8B7E007E 
	s_cbranch_execz 76                                         // 000000005BC8: BFA5004C  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:128                        // 000000005BCC: DA980080 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005BD4: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:128                     // 000000005BD8: DA9C0080 23000022 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000005BE0: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005BEC: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005BF0: 644647A3 
	ds_store_b16 v33, v35 offset:128                           // 000000005BF4: D87C0080 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005BFC: 8B7E007E 
	s_cbranch_execz 62                                         // 000000005C00: BFA5003E  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000005C04: DA9800C0 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005C0C: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:192                     // 000000005C10: DA9C00C0 23000022 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000005C18: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000005C24: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005C28: 644647A3 
	ds_store_b16 v33, v35 offset:192                           // 000000005C2C: D87C00C0 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005C34: 8B7E007E 
	s_cbranch_execz 48                                         // 000000005C38: BFA50030  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000005C3C: DA980100 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005C44: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:256                     // 000000005C48: DA9C0100 23000022 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000005C50: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000005C5C: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005C60: 644647A3 
	ds_store_b16 v33, v35 offset:256                           // 000000005C64: D87C0100 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005C6C: 8B7E007E 
	s_cbranch_execz 34                                         // 000000005C70: BFA50022  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000005C74: DA980140 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005C7C: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:320                     // 000000005C80: DA9C0140 23000022 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000005C88: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005C90: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005C94: 644647A3 
	ds_store_b16 v33, v35 offset:320                           // 000000005C98: D87C0140 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005CA0: 8B7E007E 
	s_cbranch_execz 21                                         // 000000005CA4: BFA50015  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:384                        // 000000005CA8: DA980180 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005CB0: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:384                     // 000000005CB4: DA9C0180 23000022 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000005CBC: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005CC4: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005CC8: 644647A3 
	ds_store_b16 v33, v35 offset:384                           // 000000005CCC: D87C0180 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005CD4: 8B7E007E 
	s_cbranch_execz 8                                          // 000000005CD8: BFA50008  <_amdgpu_cs_main+0x5CFC>
	ds_load_u16_d16 v35, v33 offset:448                        // 000000005CDC: DA9801C0 23000021 
	ds_load_u16_d16 v34, v34 offset:448                        // 000000005CE4: DA9801C0 22000022 
	s_waitcnt lgkmcnt(0)                                       // 000000005CEC: BF89FC07 
	v_add_f16_e32 v34, v34, v35                                // 000000005CF0: 64444722 
	ds_store_b16 v33, v34 offset:448                           // 000000005CF4: D87C01C0 00002221 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005CFC: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000005D00: BF89FC07 
	ds_store_b16 v32, v42                                      // 000000005D04: D87C0000 00002A20 
	ds_store_b16 v32, v43 offset:64                            // 000000005D0C: D87C0040 00002B20 
	ds_store_b16 v32, v44 offset:128                           // 000000005D14: D87C0080 00002C20 
	ds_store_b16 v32, v45 offset:192                           // 000000005D1C: D87C00C0 00002D20 
	ds_store_b16 v32, v46 offset:256                           // 000000005D24: D87C0100 00002E20 
	ds_store_b16 v32, v47 offset:320                           // 000000005D2C: D87C0140 00002F20 
	ds_store_b16 v32, v48 offset:384                           // 000000005D34: D87C0180 00003020 
	ds_store_b16 v32, v49 offset:448                           // 000000005D3C: D87C01C0 00003120 
	s_waitcnt lgkmcnt(0)                                       // 000000005D44: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000005D48: BE83206A 
	s_cbranch_execz 77                                         // 000000005D4C: BFA5004D  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v41 offset:2048                       // 000000005D50: DA980800 23000029 
	v_add_nc_u32_e32 v33, 0x800, v41                           // 000000005D58: 4A4252FF 00000800 
	v_add_nc_u32_e32 v34, 0xc00, v41                           // 000000005D60: 4A4452FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000005D68: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000005D74: BF89FC07 
	ds_store_b16 v41, v35 offset:3072                          // 000000005D78: D87C0C00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005D80: 8B7E007E 
	s_cbranch_execz 63                                         // 000000005D84: BFA5003F  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v33 offset:64                         // 000000005D88: DA980040 23000021 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000005D90: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005D9C: BF89FC07 
	ds_store_b16 v34, v35 offset:64                            // 000000005DA0: D87C0040 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005DA8: 8B7E007E 
	s_cbranch_execz 53                                         // 000000005DAC: BFA50035  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v33 offset:128                        // 000000005DB0: DA980080 23000021 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000005DB8: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005DC4: BF89FC07 
	ds_store_b16 v34, v35 offset:128                           // 000000005DC8: D87C0080 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005DD0: 8B7E007E 
	s_cbranch_execz 43                                         // 000000005DD4: BFA5002B  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000005DD8: DA9800C0 23000021 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000005DE0: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000005DEC: BF89FC07 
	ds_store_b16 v34, v35 offset:192                           // 000000005DF0: D87C00C0 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005DF8: 8B7E007E 
	s_cbranch_execz 33                                         // 000000005DFC: BFA50021  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000005E00: DA980100 23000021 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000005E08: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000005E14: BF89FC07 
	ds_store_b16 v34, v35 offset:256                           // 000000005E18: D87C0100 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005E20: 8B7E007E 
	s_cbranch_execz 23                                         // 000000005E24: BFA50017  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000005E28: DA980140 23000021 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000005E30: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005E38: BF89FC07 
	ds_store_b16 v34, v35 offset:320                           // 000000005E3C: D87C0140 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005E44: 8B7E007E 
	s_cbranch_execz 14                                         // 000000005E48: BFA5000E  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v35, v33 offset:384                        // 000000005E4C: DA980180 23000021 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000005E54: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005E5C: BF89FC07 
	ds_store_b16 v34, v35 offset:384                           // 000000005E60: D87C0180 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005E68: 8B7E007E 
	s_cbranch_execz 5                                          // 000000005E6C: BFA50005  <_amdgpu_cs_main+0x5E84>
	ds_load_u16_d16 v33, v33 offset:448                        // 000000005E70: DA9801C0 21000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005E78: BF89FC07 
	ds_store_b16 v34, v33 offset:448                           // 000000005E7C: D87C01C0 00002122 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005E84: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000005E88: BF89FC07 
	ds_store_b16 v32, v248                                     // 000000005E8C: D87C0000 0000F820 
	ds_store_b16 v32, v249 offset:64                           // 000000005E94: D87C0040 0000F920 
	ds_store_b16 v32, v250 offset:128                          // 000000005E9C: D87C0080 0000FA20 
	ds_store_b16 v32, v251 offset:192                          // 000000005EA4: D87C00C0 0000FB20 
	ds_store_b16 v32, v252 offset:256                          // 000000005EAC: D87C0100 0000FC20 
	ds_store_b16 v32, v253 offset:320                          // 000000005EB4: D87C0140 0000FD20 
	ds_store_b16 v32, v254 offset:384                          // 000000005EBC: D87C0180 0000FE20 
	ds_store_b16 v32, v255 offset:448                          // 000000005EC4: D87C01C0 0000FF20 
	s_waitcnt lgkmcnt(0)                                       // 000000005ECC: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000005ED0: BE83206A 
	s_cbranch_execz 108                                        // 000000005ED4: BFA5006C  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v41 offset:3072                       // 000000005ED8: DA980C00 23000029 
	s_waitcnt lgkmcnt(0)                                       // 000000005EE0: BF89FC07 
	ds_load_u16_d16_hi v35, v41 offset:2048                    // 000000005EE4: DA9C0800 23000029 
	v_add_nc_u32_e32 v33, 0xc00, v41                           // 000000005EEC: 4A4252FF 00000C00 
	v_add_nc_u32_e32 v34, 0x800, v41                           // 000000005EF4: 4A4452FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000005EFC: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000005F08: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005F0C: 644647A3 
	ds_store_b16 v41, v35 offset:3072                          // 000000005F10: D87C0C00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005F18: 8B7E007E 
	s_cbranch_execz 90                                         // 000000005F1C: BFA5005A  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:64                         // 000000005F20: DA980040 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005F28: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:64                      // 000000005F2C: DA9C0040 23000022 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000005F34: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000005F40: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005F44: 644647A3 
	ds_store_b16 v33, v35 offset:64                            // 000000005F48: D87C0040 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005F50: 8B7E007E 
	s_cbranch_execz 76                                         // 000000005F54: BFA5004C  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:128                        // 000000005F58: DA980080 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005F60: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:128                     // 000000005F64: DA9C0080 23000022 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000005F6C: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000005F78: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005F7C: 644647A3 
	ds_store_b16 v33, v35 offset:128                           // 000000005F80: D87C0080 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005F88: 8B7E007E 
	s_cbranch_execz 62                                         // 000000005F8C: BFA5003E  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000005F90: DA9800C0 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005F98: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:192                     // 000000005F9C: DA9C00C0 23000022 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000005FA4: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000005FB0: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005FB4: 644647A3 
	ds_store_b16 v33, v35 offset:192                           // 000000005FB8: D87C00C0 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005FC0: 8B7E007E 
	s_cbranch_execz 48                                         // 000000005FC4: BFA50030  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000005FC8: DA980100 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000005FD0: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:256                     // 000000005FD4: DA9C0100 23000022 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000005FDC: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000005FE8: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000005FEC: 644647A3 
	ds_store_b16 v33, v35 offset:256                           // 000000005FF0: D87C0100 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000005FF8: 8B7E007E 
	s_cbranch_execz 34                                         // 000000005FFC: BFA50022  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000006000: DA980140 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006008: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:320                     // 00000000600C: DA9C0140 23000022 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000006014: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000601C: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006020: 644647A3 
	ds_store_b16 v33, v35 offset:320                           // 000000006024: D87C0140 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000602C: 8B7E007E 
	s_cbranch_execz 21                                         // 000000006030: BFA50015  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:384                        // 000000006034: DA980180 23000021 
	s_waitcnt lgkmcnt(0)                                       // 00000000603C: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:384                     // 000000006040: DA9C0180 23000022 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000006048: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006050: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006054: 644647A3 
	ds_store_b16 v33, v35 offset:384                           // 000000006058: D87C0180 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006060: 8B7E007E 
	s_cbranch_execz 8                                          // 000000006064: BFA50008  <_amdgpu_cs_main+0x6088>
	ds_load_u16_d16 v35, v33 offset:448                        // 000000006068: DA9801C0 23000021 
	ds_load_u16_d16 v34, v34 offset:448                        // 000000006070: DA9801C0 22000022 
	s_waitcnt lgkmcnt(0)                                       // 000000006078: BF89FC07 
	v_add_f16_e32 v34, v34, v35                                // 00000000607C: 64444722 
	ds_store_b16 v33, v34 offset:448                           // 000000006080: D87C01C0 00002221 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006088: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 00000000608C: BF89FC07 
	ds_store_b16 v32, v240                                     // 000000006090: D87C0000 0000F020 
	ds_store_b16 v32, v241 offset:64                           // 000000006098: D87C0040 0000F120 
	ds_store_b16 v32, v242 offset:128                          // 0000000060A0: D87C0080 0000F220 
	ds_store_b16 v32, v243 offset:192                          // 0000000060A8: D87C00C0 0000F320 
	ds_store_b16 v32, v244 offset:256                          // 0000000060B0: D87C0100 0000F420 
	ds_store_b16 v32, v245 offset:320                          // 0000000060B8: D87C0140 0000F520 
	ds_store_b16 v32, v246 offset:384                          // 0000000060C0: D87C0180 0000F620 
	ds_store_b16 v32, v247 offset:448                          // 0000000060C8: D87C01C0 0000F720 
	s_waitcnt lgkmcnt(0)                                       // 0000000060D0: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000060D4: BE83206A 
	s_cbranch_execz 77                                         // 0000000060D8: BFA5004D  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v41 offset:2048                       // 0000000060DC: DA980800 23000029 
	v_add_nc_u32_e32 v33, 0x800, v41                           // 0000000060E4: 4A4252FF 00000800 
	v_add_nc_u32_e32 v34, 0xe00, v41                           // 0000000060EC: 4A4452FF 00000E00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000060F4: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000006100: BF89FC07 
	ds_store_b16 v41, v35 offset:3584                          // 000000006104: D87C0E00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000610C: 8B7E007E 
	s_cbranch_execz 63                                         // 000000006110: BFA5003F  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v33 offset:64                         // 000000006114: DA980040 23000021 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 00000000611C: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006128: BF89FC07 
	ds_store_b16 v34, v35 offset:64                            // 00000000612C: D87C0040 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006134: 8B7E007E 
	s_cbranch_execz 53                                         // 000000006138: BFA50035  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v33 offset:128                        // 00000000613C: DA980080 23000021 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000006144: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006150: BF89FC07 
	ds_store_b16 v34, v35 offset:128                           // 000000006154: D87C0080 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000615C: 8B7E007E 
	s_cbranch_execz 43                                         // 000000006160: BFA5002B  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000006164: DA9800C0 23000021 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000616C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000006178: BF89FC07 
	ds_store_b16 v34, v35 offset:192                           // 00000000617C: D87C00C0 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006184: 8B7E007E 
	s_cbranch_execz 33                                         // 000000006188: BFA50021  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v33 offset:256                        // 00000000618C: DA980100 23000021 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000006194: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000061A0: BF89FC07 
	ds_store_b16 v34, v35 offset:256                           // 0000000061A4: D87C0100 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000061AC: 8B7E007E 
	s_cbranch_execz 23                                         // 0000000061B0: BFA50017  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v33 offset:320                        // 0000000061B4: DA980140 23000021 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000061BC: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000061C4: BF89FC07 
	ds_store_b16 v34, v35 offset:320                           // 0000000061C8: D87C0140 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000061D0: 8B7E007E 
	s_cbranch_execz 14                                         // 0000000061D4: BFA5000E  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v35, v33 offset:384                        // 0000000061D8: DA980180 23000021 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000061E0: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000061E8: BF89FC07 
	ds_store_b16 v34, v35 offset:384                           // 0000000061EC: D87C0180 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000061F4: 8B7E007E 
	s_cbranch_execz 5                                          // 0000000061F8: BFA50005  <_amdgpu_cs_main+0x6210>
	ds_load_u16_d16 v33, v33 offset:448                        // 0000000061FC: DA9801C0 21000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006204: BF89FC07 
	ds_store_b16 v34, v33 offset:448                           // 000000006208: D87C01C0 00002122 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006210: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000006214: BF89FC07 
	ds_store_b16 v32, v81                                      // 000000006218: D87C0000 00005120 
	ds_store_b16 v32, v82 offset:64                            // 000000006220: D87C0040 00005220 
	ds_store_b16 v32, v83 offset:128                           // 000000006228: D87C0080 00005320 
	ds_store_b16 v32, v84 offset:192                           // 000000006230: D87C00C0 00005420 
	ds_store_b16 v32, v85 offset:256                           // 000000006238: D87C0100 00005520 
	ds_store_b16 v32, v86 offset:320                           // 000000006240: D87C0140 00005620 
	ds_store_b16 v32, v87 offset:384                           // 000000006248: D87C0180 00005720 
	ds_store_b16 v32, v88 offset:448                           // 000000006250: D87C01C0 00005820 
	s_waitcnt lgkmcnt(0)                                       // 000000006258: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 00000000625C: BE83206A 
	s_cbranch_execz 108                                        // 000000006260: BFA5006C  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v41 offset:3584                       // 000000006264: DA980E00 23000029 
	s_waitcnt lgkmcnt(0)                                       // 00000000626C: BF89FC07 
	ds_load_u16_d16_hi v35, v41 offset:2048                    // 000000006270: DA9C0800 23000029 
	v_add_nc_u32_e32 v33, 0xe00, v41                           // 000000006278: 4A4252FF 00000E00 
	v_add_nc_u32_e32 v34, 0x800, v41                           // 000000006280: 4A4452FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006288: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000006294: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006298: 644647A3 
	ds_store_b16 v41, v35 offset:3584                          // 00000000629C: D87C0E00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000062A4: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000062A8: BFA5005A  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:64                         // 0000000062AC: DA980040 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000062B4: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:64                      // 0000000062B8: DA9C0040 23000022 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000062C0: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000062CC: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000062D0: 644647A3 
	ds_store_b16 v33, v35 offset:64                            // 0000000062D4: D87C0040 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000062DC: 8B7E007E 
	s_cbranch_execz 76                                         // 0000000062E0: BFA5004C  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:128                        // 0000000062E4: DA980080 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000062EC: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:128                     // 0000000062F0: DA9C0080 23000022 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000062F8: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006304: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006308: 644647A3 
	ds_store_b16 v33, v35 offset:128                           // 00000000630C: D87C0080 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006314: 8B7E007E 
	s_cbranch_execz 62                                         // 000000006318: BFA5003E  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:192                        // 00000000631C: DA9800C0 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006324: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:192                     // 000000006328: DA9C00C0 23000022 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000006330: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 00000000633C: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006340: 644647A3 
	ds_store_b16 v33, v35 offset:192                           // 000000006344: D87C00C0 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000634C: 8B7E007E 
	s_cbranch_execz 48                                         // 000000006350: BFA50030  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000006354: DA980100 23000021 
	s_waitcnt lgkmcnt(0)                                       // 00000000635C: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:256                     // 000000006360: DA9C0100 23000022 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000006368: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000006374: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006378: 644647A3 
	ds_store_b16 v33, v35 offset:256                           // 00000000637C: D87C0100 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006384: 8B7E007E 
	s_cbranch_execz 34                                         // 000000006388: BFA50022  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:320                        // 00000000638C: DA980140 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006394: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:320                     // 000000006398: DA9C0140 23000022 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000063A0: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000063A8: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000063AC: 644647A3 
	ds_store_b16 v33, v35 offset:320                           // 0000000063B0: D87C0140 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000063B8: 8B7E007E 
	s_cbranch_execz 21                                         // 0000000063BC: BFA50015  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:384                        // 0000000063C0: DA980180 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000063C8: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:384                     // 0000000063CC: DA9C0180 23000022 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000063D4: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000063DC: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000063E0: 644647A3 
	ds_store_b16 v33, v35 offset:384                           // 0000000063E4: D87C0180 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000063EC: 8B7E007E 
	s_cbranch_execz 8                                          // 0000000063F0: BFA50008  <_amdgpu_cs_main+0x6414>
	ds_load_u16_d16 v35, v33 offset:448                        // 0000000063F4: DA9801C0 23000021 
	ds_load_u16_d16 v34, v34 offset:448                        // 0000000063FC: DA9801C0 22000022 
	s_waitcnt lgkmcnt(0)                                       // 000000006404: BF89FC07 
	v_add_f16_e32 v34, v34, v35                                // 000000006408: 64444722 
	ds_store_b16 v33, v34 offset:448                           // 00000000640C: D87C01C0 00002221 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006414: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000006418: BF89FC07 
	ds_store_b16 v32, v89                                      // 00000000641C: D87C0000 00005920 
	ds_store_b16 v32, v90 offset:64                            // 000000006424: D87C0040 00005A20 
	ds_store_b16 v32, v91 offset:128                           // 00000000642C: D87C0080 00005B20 
	ds_store_b16 v32, v92 offset:192                           // 000000006434: D87C00C0 00005C20 
	ds_store_b16 v32, v93 offset:256                           // 00000000643C: D87C0100 00005D20 
	ds_store_b16 v32, v94 offset:320                           // 000000006444: D87C0140 00005E20 
	ds_store_b16 v32, v95 offset:384                           // 00000000644C: D87C0180 00005F20 
	ds_store_b16 v32, v96 offset:448                           // 000000006454: D87C01C0 00006020 
	s_waitcnt lgkmcnt(0)                                       // 00000000645C: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000006460: BE83206A 
	s_cbranch_execz 77                                         // 000000006464: BFA5004D  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v41 offset:2048                       // 000000006468: DA980800 23000029 
	v_add_nc_u32_e32 v33, 0x800, v41                           // 000000006470: 4A4252FF 00000800 
	v_add_nc_u32_e32 v34, 0x1000, v41                          // 000000006478: 4A4452FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006480: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 00000000648C: BF89FC07 
	ds_store_b16 v41, v35 offset:4096                          // 000000006490: D87C1000 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006498: 8B7E007E 
	s_cbranch_execz 63                                         // 00000000649C: BFA5003F  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v33 offset:64                         // 0000000064A0: DA980040 23000021 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000064A8: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000064B4: BF89FC07 
	ds_store_b16 v34, v35 offset:64                            // 0000000064B8: D87C0040 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000064C0: 8B7E007E 
	s_cbranch_execz 53                                         // 0000000064C4: BFA50035  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v33 offset:128                        // 0000000064C8: DA980080 23000021 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000064D0: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000064DC: BF89FC07 
	ds_store_b16 v34, v35 offset:128                           // 0000000064E0: D87C0080 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000064E8: 8B7E007E 
	s_cbranch_execz 43                                         // 0000000064EC: BFA5002B  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v33 offset:192                        // 0000000064F0: DA9800C0 23000021 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 0000000064F8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000006504: BF89FC07 
	ds_store_b16 v34, v35 offset:192                           // 000000006508: D87C00C0 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006510: 8B7E007E 
	s_cbranch_execz 33                                         // 000000006514: BFA50021  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000006518: DA980100 23000021 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000006520: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 00000000652C: BF89FC07 
	ds_store_b16 v34, v35 offset:256                           // 000000006530: D87C0100 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006538: 8B7E007E 
	s_cbranch_execz 23                                         // 00000000653C: BFA50017  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000006540: DA980140 23000021 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000006548: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006550: BF89FC07 
	ds_store_b16 v34, v35 offset:320                           // 000000006554: D87C0140 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000655C: 8B7E007E 
	s_cbranch_execz 14                                         // 000000006560: BFA5000E  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v35, v33 offset:384                        // 000000006564: DA980180 23000021 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 00000000656C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006574: BF89FC07 
	ds_store_b16 v34, v35 offset:384                           // 000000006578: D87C0180 00002322 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006580: 8B7E007E 
	s_cbranch_execz 5                                          // 000000006584: BFA50005  <_amdgpu_cs_main+0x659C>
	ds_load_u16_d16 v33, v33 offset:448                        // 000000006588: DA9801C0 21000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006590: BF89FC07 
	ds_store_b16 v34, v33 offset:448                           // 000000006594: D87C01C0 00002122 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000659C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000065A0: BF89FC07 
	ds_store_b16 v32, v97                                      // 0000000065A4: D87C0000 00006120 
	ds_store_b16 v32, v98 offset:64                            // 0000000065AC: D87C0040 00006220 
	ds_store_b16 v32, v99 offset:128                           // 0000000065B4: D87C0080 00006320 
	ds_store_b16 v32, v100 offset:192                          // 0000000065BC: D87C00C0 00006420 
	ds_store_b16 v32, v101 offset:256                          // 0000000065C4: D87C0100 00006520 
	ds_store_b16 v32, v102 offset:320                          // 0000000065CC: D87C0140 00006620 
	ds_store_b16 v32, v103 offset:384                          // 0000000065D4: D87C0180 00006720 
	ds_store_b16 v32, v104 offset:448                          // 0000000065DC: D87C01C0 00006820 
	s_waitcnt lgkmcnt(0)                                       // 0000000065E4: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000065E8: BE83206A 
	s_cbranch_execz 108                                        // 0000000065EC: BFA5006C  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v41 offset:4096                       // 0000000065F0: DA981000 23000029 
	s_waitcnt lgkmcnt(0)                                       // 0000000065F8: BF89FC07 
	ds_load_u16_d16_hi v35, v41 offset:2048                    // 0000000065FC: DA9C0800 23000029 
	v_add_nc_u32_e32 v33, 0x1000, v41                          // 000000006604: 4A4252FF 00001000 
	v_add_nc_u32_e32 v34, 0x800, v41                           // 00000000660C: 4A4452FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006614: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000006620: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006624: 644647A3 
	ds_store_b16 v41, v35 offset:4096                          // 000000006628: D87C1000 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006630: 8B7E007E 
	s_cbranch_execz 90                                         // 000000006634: BFA5005A  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:64                         // 000000006638: DA980040 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006640: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:64                      // 000000006644: DA9C0040 23000022 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 00000000664C: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006658: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 00000000665C: 644647A3 
	ds_store_b16 v33, v35 offset:64                            // 000000006660: D87C0040 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006668: 8B7E007E 
	s_cbranch_execz 76                                         // 00000000666C: BFA5004C  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:128                        // 000000006670: DA980080 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006678: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:128                     // 00000000667C: DA9C0080 23000022 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000006684: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006690: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006694: 644647A3 
	ds_store_b16 v33, v35 offset:128                           // 000000006698: D87C0080 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000066A0: 8B7E007E 
	s_cbranch_execz 62                                         // 0000000066A4: BFA5003E  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:192                        // 0000000066A8: DA9800C0 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000066B0: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:192                     // 0000000066B4: DA9C00C0 23000022 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 0000000066BC: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 0000000066C8: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000066CC: 644647A3 
	ds_store_b16 v33, v35 offset:192                           // 0000000066D0: D87C00C0 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000066D8: 8B7E007E 
	s_cbranch_execz 48                                         // 0000000066DC: BFA50030  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:256                        // 0000000066E0: DA980100 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000066E8: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:256                     // 0000000066EC: DA9C0100 23000022 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000066F4: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000006700: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006704: 644647A3 
	ds_store_b16 v33, v35 offset:256                           // 000000006708: D87C0100 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006710: 8B7E007E 
	s_cbranch_execz 34                                         // 000000006714: BFA50022  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000006718: DA980140 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006720: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:320                     // 000000006724: DA9C0140 23000022 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000672C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006734: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006738: 644647A3 
	ds_store_b16 v33, v35 offset:320                           // 00000000673C: D87C0140 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006744: 8B7E007E 
	s_cbranch_execz 21                                         // 000000006748: BFA50015  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:384                        // 00000000674C: DA980180 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006754: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:384                     // 000000006758: DA9C0180 23000022 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000006760: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006768: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 00000000676C: 644647A3 
	ds_store_b16 v33, v35 offset:384                           // 000000006770: D87C0180 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006778: 8B7E007E 
	s_cbranch_execz 8                                          // 00000000677C: BFA50008  <_amdgpu_cs_main+0x67A0>
	ds_load_u16_d16 v35, v33 offset:448                        // 000000006780: DA9801C0 23000021 
	ds_load_u16_d16 v34, v34 offset:448                        // 000000006788: DA9801C0 22000022 
	s_waitcnt lgkmcnt(0)                                       // 000000006790: BF89FC07 
	v_add_f16_e32 v34, v34, v35                                // 000000006794: 64444722 
	ds_store_b16 v33, v34 offset:448                           // 000000006798: D87C01C0 00002221 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000067A0: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000067A4: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000067A8: BE83206A 
	s_cbranch_execz 108                                        // 0000000067AC: BFA5006C  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v41 offset:2560                       // 0000000067B0: DA980A00 23000029 
	s_waitcnt lgkmcnt(0)                                       // 0000000067B8: BF89FC07 
	ds_load_u16_d16_hi v35, v41 offset:3072                    // 0000000067BC: DA9C0C00 23000029 
	v_add_nc_u32_e32 v33, 0xa00, v41                           // 0000000067C4: 4A4252FF 00000A00 
	v_add_nc_u32_e32 v34, 0xc00, v41                           // 0000000067CC: 4A4452FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000067D4: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000067E0: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000067E4: 644647A3 
	ds_store_b16 v41, v35 offset:2560                          // 0000000067E8: D87C0A00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000067F0: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000067F4: BFA5005A  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:64                         // 0000000067F8: DA980040 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006800: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:64                      // 000000006804: DA9C0040 23000022 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 00000000680C: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006818: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 00000000681C: 644647A3 
	ds_store_b16 v33, v35 offset:64                            // 000000006820: D87C0040 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006828: 8B7E007E 
	s_cbranch_execz 76                                         // 00000000682C: BFA5004C  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:128                        // 000000006830: DA980080 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006838: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:128                     // 00000000683C: DA9C0080 23000022 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000006844: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006850: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006854: 644647A3 
	ds_store_b16 v33, v35 offset:128                           // 000000006858: D87C0080 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006860: 8B7E007E 
	s_cbranch_execz 62                                         // 000000006864: BFA5003E  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000006868: DA9800C0 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006870: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:192                     // 000000006874: DA9C00C0 23000022 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000687C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000006888: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 00000000688C: 644647A3 
	ds_store_b16 v33, v35 offset:192                           // 000000006890: D87C00C0 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006898: 8B7E007E 
	s_cbranch_execz 48                                         // 00000000689C: BFA50030  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:256                        // 0000000068A0: DA980100 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000068A8: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:256                     // 0000000068AC: DA9C0100 23000022 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000068B4: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000068C0: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000068C4: 644647A3 
	ds_store_b16 v33, v35 offset:256                           // 0000000068C8: D87C0100 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000068D0: 8B7E007E 
	s_cbranch_execz 34                                         // 0000000068D4: BFA50022  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:320                        // 0000000068D8: DA980140 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000068E0: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:320                     // 0000000068E4: DA9C0140 23000022 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000068EC: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000068F4: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000068F8: 644647A3 
	ds_store_b16 v33, v35 offset:320                           // 0000000068FC: D87C0140 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006904: 8B7E007E 
	s_cbranch_execz 21                                         // 000000006908: BFA50015  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:384                        // 00000000690C: DA980180 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006914: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:384                     // 000000006918: DA9C0180 23000022 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000006920: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006928: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 00000000692C: 644647A3 
	ds_store_b16 v33, v35 offset:384                           // 000000006930: D87C0180 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006938: 8B7E007E 
	s_cbranch_execz 8                                          // 00000000693C: BFA50008  <_amdgpu_cs_main+0x6960>
	ds_load_u16_d16 v35, v33 offset:448                        // 000000006940: DA9801C0 23000021 
	ds_load_u16_d16 v34, v34 offset:448                        // 000000006948: DA9801C0 22000022 
	s_waitcnt lgkmcnt(0)                                       // 000000006950: BF89FC07 
	v_add_f16_e32 v34, v34, v35                                // 000000006954: 64444722 
	ds_store_b16 v33, v34 offset:448                           // 000000006958: D87C01C0 00002221 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006960: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000006964: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000006968: BE83206A 
	s_cbranch_execz 108                                        // 00000000696C: BFA5006C  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v41 offset:3584                       // 000000006970: DA980E00 23000029 
	s_waitcnt lgkmcnt(0)                                       // 000000006978: BF89FC07 
	ds_load_u16_d16_hi v35, v41 offset:4096                    // 00000000697C: DA9C1000 23000029 
	v_add_nc_u32_e32 v33, 0xe00, v41                           // 000000006984: 4A4252FF 00000E00 
	v_add_nc_u32_e32 v34, 0x1000, v41                          // 00000000698C: 4A4452FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006994: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000069A0: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000069A4: 644647A3 
	ds_store_b16 v41, v35 offset:3584                          // 0000000069A8: D87C0E00 00002329 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000069B0: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000069B4: BFA5005A  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:64                         // 0000000069B8: DA980040 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000069C0: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:64                      // 0000000069C4: DA9C0040 23000022 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000069CC: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000069D8: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 0000000069DC: 644647A3 
	ds_store_b16 v33, v35 offset:64                            // 0000000069E0: D87C0040 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000069E8: 8B7E007E 
	s_cbranch_execz 76                                         // 0000000069EC: BFA5004C  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:128                        // 0000000069F0: DA980080 23000021 
	s_waitcnt lgkmcnt(0)                                       // 0000000069F8: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:128                     // 0000000069FC: DA9C0080 23000022 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000006A04: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006A10: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006A14: 644647A3 
	ds_store_b16 v33, v35 offset:128                           // 000000006A18: D87C0080 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006A20: 8B7E007E 
	s_cbranch_execz 62                                         // 000000006A24: BFA5003E  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:192                        // 000000006A28: DA9800C0 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006A30: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:192                     // 000000006A34: DA9C00C0 23000022 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000006A3C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000006A48: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006A4C: 644647A3 
	ds_store_b16 v33, v35 offset:192                           // 000000006A50: D87C00C0 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006A58: 8B7E007E 
	s_cbranch_execz 48                                         // 000000006A5C: BFA50030  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:256                        // 000000006A60: DA980100 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006A68: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:256                     // 000000006A6C: DA9C0100 23000022 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000006A74: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000006A80: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006A84: 644647A3 
	ds_store_b16 v33, v35 offset:256                           // 000000006A88: D87C0100 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006A90: 8B7E007E 
	s_cbranch_execz 34                                         // 000000006A94: BFA50022  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:320                        // 000000006A98: DA980140 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006AA0: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:320                     // 000000006AA4: DA9C0140 23000022 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000006AAC: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006AB4: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006AB8: 644647A3 
	ds_store_b16 v33, v35 offset:320                           // 000000006ABC: D87C0140 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006AC4: 8B7E007E 
	s_cbranch_execz 21                                         // 000000006AC8: BFA50015  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:384                        // 000000006ACC: DA980180 23000021 
	s_waitcnt lgkmcnt(0)                                       // 000000006AD4: BF89FC07 
	ds_load_u16_d16_hi v35, v34 offset:384                     // 000000006AD8: DA9C0180 23000022 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000006AE0: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006AE8: BF89FC07 
	v_add_f16_e32 v35, v163/*Invalid register, operand has 'VS_32_Lo128' register class*/, v35// 000000006AEC: 644647A3 
	ds_store_b16 v33, v35 offset:384                           // 000000006AF0: D87C0180 00002321 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006AF8: 8B7E007E 
	s_cbranch_execz 8                                          // 000000006AFC: BFA50008  <_amdgpu_cs_main+0x6B20>
	ds_load_u16_d16 v35, v33 offset:448                        // 000000006B00: DA9801C0 23000021 
	ds_load_u16_d16 v34, v34 offset:448                        // 000000006B08: DA9801C0 22000022 
	s_waitcnt lgkmcnt(0)                                       // 000000006B10: BF89FC07 
	v_add_f16_e32 v34, v34, v35                                // 000000006B14: 64444722 
	ds_store_b16 v33, v34 offset:448                           // 000000006B18: D87C01C0 00002221 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006B20: 8C7E037E 
	v_subrev_nc_u32_e32 v33, 32, v129                          // 000000006B24: 4E4302A0 
	v_lshrrev_b32_e32 v34, 4, v129                             // 000000006B28: 32450284 
	s_waitcnt lgkmcnt(0)                                       // 000000006B2C: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 000000006B30: BE84206A 
	s_cbranch_execz 49                                         // 000000006B34: BFA50031  <_amdgpu_cs_main+0x6BFC>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006B38: BF870001 
	v_add_nc_u32_e32 v35, s11, v34                             // 000000006B3C: 4A46440B 
	v_or_b32_e32 v36, s10, v128                                // 000000006B40: 3849000A 
	v_add_nc_u32_e32 v37, 0xa00, v41                           // 000000006B44: 4A4A52FF 00000A00 
	v_add_nc_u32_e32 v38, 0xe00, v41                           // 000000006B4C: 4A4C52FF 00000E00 
	s_lshl_b32 s5, s6, 3                                       // 000000006B54: 84058306 
	v_mad_u64_u32 v[39:40], null, s6, v35, s[10:11]            // 000000006B58: D6FE7C27 002A4606 
	v_cmp_gt_u32_e64 s0, s2, v36                               // 000000006B60: D44C0000 00024802 
	v_subrev_nc_u32_e32 v36, 32, v129                          // 000000006B68: 4E4902A0 
	s_mov_b32 s7, 0                                            // 000000006B6C: BE870080 
	v_add_lshl_u32 v39, v39, v128, 2                           // 000000006B70: D6470027 020B0127 
	s_set_inst_prefetch_distance 0x1                           // 000000006B78: BF840001 
	s_branch 13                                                // 000000006B7C: BFA0000D  <_amdgpu_cs_main+0x6BB4>
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006B80: 8C7E037E 
	v_add_nc_u32_e32 v36, 32, v36                              // 000000006B84: 4A4848A0 
	v_add_nc_u32_e32 v37, 64, v37                              // 000000006B88: 4A4A4AC0 
	v_add_nc_u32_e32 v38, 64, v38                              // 000000006B8C: 4A4C4CC0 
	v_add_nc_u32_e32 v39, s5, v39                              // 000000006B90: 4A4E4E05 
	v_add_nc_u32_e32 v35, 2, v35                               // 000000006B94: 4A464682 
	v_cmp_lt_u32_e64 s3, 0xdf, v36                             // 000000006B98: D4490003 000248FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000006BA4: 8C070703 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006BA8: BF870009 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 000000006BAC: 917E077E 
	s_cbranch_execz 18                                         // 000000006BB0: BFA50012  <_amdgpu_cs_main+0x6BFC>
	v_cmp_gt_u32_e64 s3, s1, v35                               // 000000006BB4: D44C0003 00024601 
	s_and_b32 s8, s3, s0                                       // 000000006BBC: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006BC0: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000006BC4: BE832008 
	s_cbranch_execz 65517                                      // 000000006BC8: BFA5FFED  <_amdgpu_cs_main+0x6B80>
	ds_load_u16_d16 v40, v37                                   // 000000006BCC: DA980000 28000025 
	s_waitcnt lgkmcnt(0)                                       // 000000006BD4: BF89FC07 
	ds_load_u16_d16_hi v40, v38                                // 000000006BD8: DA9C0000 28000026 
	s_waitcnt lgkmcnt(0)                                       // 000000006BE0: BF89FC07 
	v_add_f16_e32 v40, v168/*Invalid register, operand has 'VS_32_Lo128' register class*/, v40// 000000006BE4: 645051A8 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006BE8: BF870001 
	v_cvt_f32_f16_e32 v40, v40                                 // 000000006BEC: 7E501728 
	buffer_store_b32 v40, v39, s[20:23], 0 offen               // 000000006BF0: E0680000 80452827 
	s_branch 65505                                             // 000000006BF8: BFA0FFE1  <_amdgpu_cs_main+0x6B80>
	s_set_inst_prefetch_distance 0x2                           // 000000006BFC: BF840002 
	s_or_b32 exec_lo, exec_lo, s4                              // 000000006C00: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 000000006C04: BC7C0000 
	ds_store_b16_d16_hi v32, v8                                // 000000006C08: DA840000 00000820 
	ds_store_b16_d16_hi v32, v9 offset:64                      // 000000006C10: DA840040 00000920 
	ds_store_b16_d16_hi v32, v10 offset:128                    // 000000006C18: DA840080 00000A20 
	ds_store_b16_d16_hi v32, v11 offset:192                    // 000000006C20: DA8400C0 00000B20 
	ds_store_b16_d16_hi v32, v12 offset:256                    // 000000006C28: DA840100 00000C20 
	ds_store_b16_d16_hi v32, v13 offset:320                    // 000000006C30: DA840140 00000D20 
	ds_store_b16_d16_hi v32, v14 offset:384                    // 000000006C38: DA840180 00000E20 
	ds_store_b16_d16_hi v32, v15 offset:448                    // 000000006C40: DA8401C0 00000F20 
	s_waitcnt lgkmcnt(0)                                       // 000000006C48: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000006C4C: BE83206A 
	s_cbranch_execz 77                                         // 000000006C50: BFA5004D  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v41 offset:2048                       // 000000006C54: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 000000006C5C: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0xa00, v41                            // 000000006C64: 4A1252FF 00000A00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006C6C: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000006C78: BF89FC07 
	ds_store_b16 v41, v10 offset:2560                          // 000000006C7C: D87C0A00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006C84: 8B7E007E 
	s_cbranch_execz 63                                         // 000000006C88: BFA5003F  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000006C8C: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000006C94: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006CA0: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 000000006CA4: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006CAC: 8B7E007E 
	s_cbranch_execz 53                                         // 000000006CB0: BFA50035  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000006CB4: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000006CBC: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006CC8: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 000000006CCC: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006CD4: 8B7E007E 
	s_cbranch_execz 43                                         // 000000006CD8: BFA5002B  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000006CDC: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000006CE4: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000006CF0: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 000000006CF4: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006CFC: 8B7E007E 
	s_cbranch_execz 33                                         // 000000006D00: BFA50021  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000006D04: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000006D0C: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000006D18: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 000000006D1C: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006D24: 8B7E007E 
	s_cbranch_execz 23                                         // 000000006D28: BFA50017  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000006D2C: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000006D34: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006D3C: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 000000006D40: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006D48: 8B7E007E 
	s_cbranch_execz 14                                         // 000000006D4C: BFA5000E  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000006D50: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000006D58: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006D60: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 000000006D64: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006D6C: 8B7E007E 
	s_cbranch_execz 5                                          // 000000006D70: BFA50005  <_amdgpu_cs_main+0x6D88>
	ds_load_u16_d16 v8, v8 offset:448                          // 000000006D74: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006D7C: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 000000006D80: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006D88: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000006D8C: BF89FC07 
	ds_store_b16_d16_hi v32, v16                               // 000000006D90: DA840000 00001020 
	ds_store_b16_d16_hi v32, v17 offset:64                     // 000000006D98: DA840040 00001120 
	ds_store_b16_d16_hi v32, v18 offset:128                    // 000000006DA0: DA840080 00001220 
	ds_store_b16_d16_hi v32, v19 offset:192                    // 000000006DA8: DA8400C0 00001320 
	ds_store_b16_d16_hi v32, v20 offset:256                    // 000000006DB0: DA840100 00001420 
	ds_store_b16_d16_hi v32, v21 offset:320                    // 000000006DB8: DA840140 00001520 
	ds_store_b16_d16_hi v32, v22 offset:384                    // 000000006DC0: DA840180 00001620 
	ds_store_b16_d16_hi v32, v23 offset:448                    // 000000006DC8: DA8401C0 00001720 
	s_waitcnt lgkmcnt(0)                                       // 000000006DD0: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000006DD4: BE83206A 
	s_cbranch_execz 108                                        // 000000006DD8: BFA5006C  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v41 offset:2560                       // 000000006DDC: DA980A00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000006DE4: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 000000006DE8: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0xa00, v41                            // 000000006DF0: 4A1052FF 00000A00 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 000000006DF8: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006E00: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000006E0C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006E10: 6414158A 
	ds_store_b16 v41, v10 offset:2560                          // 000000006E14: D87C0A00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006E1C: 8B7E007E 
	s_cbranch_execz 90                                         // 000000006E20: BFA5005A  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000006E24: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006E2C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000006E30: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000006E38: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006E44: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006E48: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000006E4C: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006E54: 8B7E007E 
	s_cbranch_execz 76                                         // 000000006E58: BFA5004C  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000006E5C: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006E64: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000006E68: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000006E70: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006E7C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006E80: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000006E84: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006E8C: 8B7E007E 
	s_cbranch_execz 62                                         // 000000006E90: BFA5003E  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000006E94: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006E9C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000006EA0: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000006EA8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000006EB4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006EB8: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000006EBC: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006EC4: 8B7E007E 
	s_cbranch_execz 48                                         // 000000006EC8: BFA50030  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000006ECC: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006ED4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000006ED8: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000006EE0: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000006EEC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006EF0: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000006EF4: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006EFC: 8B7E007E 
	s_cbranch_execz 34                                         // 000000006F00: BFA50022  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000006F04: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006F0C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000006F10: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000006F18: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000006F20: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006F24: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000006F28: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006F30: 8B7E007E 
	s_cbranch_execz 21                                         // 000000006F34: BFA50015  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000006F38: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000006F40: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000006F44: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000006F4C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000006F54: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000006F58: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000006F5C: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000006F64: 8B7E007E 
	s_cbranch_execz 8                                          // 000000006F68: BFA50008  <_amdgpu_cs_main+0x6F8C>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000006F6C: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000006F74: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000006F7C: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000006F80: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000006F84: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006F8C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000006F90: BF89FC07 
	ds_store_b16_d16_hi v32, v42                               // 000000006F94: DA840000 00002A20 
	ds_store_b16_d16_hi v32, v43 offset:64                     // 000000006F9C: DA840040 00002B20 
	ds_store_b16_d16_hi v32, v44 offset:128                    // 000000006FA4: DA840080 00002C20 
	ds_store_b16_d16_hi v32, v45 offset:192                    // 000000006FAC: DA8400C0 00002D20 
	ds_store_b16_d16_hi v32, v46 offset:256                    // 000000006FB4: DA840100 00002E20 
	ds_store_b16_d16_hi v32, v47 offset:320                    // 000000006FBC: DA840140 00002F20 
	ds_store_b16_d16_hi v32, v48 offset:384                    // 000000006FC4: DA840180 00003020 
	ds_store_b16_d16_hi v32, v49 offset:448                    // 000000006FCC: DA8401C0 00003120 
	s_waitcnt lgkmcnt(0)                                       // 000000006FD4: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000006FD8: BE83206A 
	s_cbranch_execz 77                                         // 000000006FDC: BFA5004D  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v41 offset:2048                       // 000000006FE0: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 000000006FE8: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0xc00, v41                            // 000000006FF0: 4A1252FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000006FF8: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007004: BF89FC07 
	ds_store_b16 v41, v10 offset:3072                          // 000000007008: D87C0C00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007010: 8B7E007E 
	s_cbranch_execz 63                                         // 000000007014: BFA5003F  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000007018: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000007020: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000702C: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 000000007030: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007038: 8B7E007E 
	s_cbranch_execz 53                                         // 00000000703C: BFA50035  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007040: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007048: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007054: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 000000007058: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007060: 8B7E007E 
	s_cbranch_execz 43                                         // 000000007064: BFA5002B  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007068: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000007070: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 00000000707C: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 000000007080: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007088: 8B7E007E 
	s_cbranch_execz 33                                         // 00000000708C: BFA50021  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000007090: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000007098: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000070A4: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 0000000070A8: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000070B0: 8B7E007E 
	s_cbranch_execz 23                                         // 0000000070B4: BFA50017  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v8 offset:320                         // 0000000070B8: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000070C0: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000070C8: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 0000000070CC: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000070D4: 8B7E007E 
	s_cbranch_execz 14                                         // 0000000070D8: BFA5000E  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v10, v8 offset:384                         // 0000000070DC: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000070E4: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000070EC: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 0000000070F0: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000070F8: 8B7E007E 
	s_cbranch_execz 5                                          // 0000000070FC: BFA50005  <_amdgpu_cs_main+0x7114>
	ds_load_u16_d16 v8, v8 offset:448                          // 000000007100: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007108: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 00000000710C: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007114: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000007118: BF89FC07 
	ds_store_b16_d16_hi v32, v248                              // 00000000711C: DA840000 0000F820 
	ds_store_b16_d16_hi v32, v249 offset:64                    // 000000007124: DA840040 0000F920 
	ds_store_b16_d16_hi v32, v250 offset:128                   // 00000000712C: DA840080 0000FA20 
	ds_store_b16_d16_hi v32, v251 offset:192                   // 000000007134: DA8400C0 0000FB20 
	ds_store_b16_d16_hi v32, v252 offset:256                   // 00000000713C: DA840100 0000FC20 
	ds_store_b16_d16_hi v32, v253 offset:320                   // 000000007144: DA840140 0000FD20 
	ds_store_b16_d16_hi v32, v254 offset:384                   // 00000000714C: DA840180 0000FE20 
	ds_store_b16_d16_hi v32, v255 offset:448                   // 000000007154: DA8401C0 0000FF20 
	s_waitcnt lgkmcnt(0)                                       // 00000000715C: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000007160: BE83206A 
	s_cbranch_execz 108                                        // 000000007164: BFA5006C  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v41 offset:3072                       // 000000007168: DA980C00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000007170: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 000000007174: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0xc00, v41                            // 00000000717C: 4A1052FF 00000C00 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 000000007184: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 00000000718C: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007198: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000719C: 6414158A 
	ds_store_b16 v41, v10 offset:3072                          // 0000000071A0: D87C0C00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000071A8: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000071AC: BFA5005A  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:64                          // 0000000071B0: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000071B8: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 0000000071BC: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000071C4: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000071D0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000071D4: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 0000000071D8: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000071E0: 8B7E007E 
	s_cbranch_execz 76                                         // 0000000071E4: BFA5004C  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:128                         // 0000000071E8: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000071F0: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 0000000071F4: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000071FC: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007208: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000720C: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000007210: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007218: 8B7E007E 
	s_cbranch_execz 62                                         // 00000000721C: BFA5003E  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007220: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007228: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 00000000722C: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000007234: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007240: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007244: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000007248: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007250: 8B7E007E 
	s_cbranch_execz 48                                         // 000000007254: BFA50030  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000007258: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007260: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000007264: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 00000000726C: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007278: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000727C: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000007280: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007288: 8B7E007E 
	s_cbranch_execz 34                                         // 00000000728C: BFA50022  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000007290: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007298: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 00000000729C: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000072A4: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000072AC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000072B0: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 0000000072B4: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000072BC: 8B7E007E 
	s_cbranch_execz 21                                         // 0000000072C0: BFA50015  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:384                         // 0000000072C4: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000072CC: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 0000000072D0: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000072D8: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000072E0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000072E4: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 0000000072E8: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000072F0: 8B7E007E 
	s_cbranch_execz 8                                          // 0000000072F4: BFA50008  <_amdgpu_cs_main+0x7318>
	ds_load_u16_d16 v10, v8 offset:448                         // 0000000072F8: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000007300: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000007308: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 00000000730C: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000007310: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007318: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 00000000731C: BF89FC07 
	ds_store_b16_d16_hi v32, v240                              // 000000007320: DA840000 0000F020 
	ds_store_b16_d16_hi v32, v241 offset:64                    // 000000007328: DA840040 0000F120 
	ds_store_b16_d16_hi v32, v242 offset:128                   // 000000007330: DA840080 0000F220 
	ds_store_b16_d16_hi v32, v243 offset:192                   // 000000007338: DA8400C0 0000F320 
	ds_store_b16_d16_hi v32, v244 offset:256                   // 000000007340: DA840100 0000F420 
	ds_store_b16_d16_hi v32, v245 offset:320                   // 000000007348: DA840140 0000F520 
	ds_store_b16_d16_hi v32, v246 offset:384                   // 000000007350: DA840180 0000F620 
	ds_store_b16_d16_hi v32, v247 offset:448                   // 000000007358: DA8401C0 0000F720 
	s_waitcnt lgkmcnt(0)                                       // 000000007360: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000007364: BE83206A 
	s_cbranch_execz 77                                         // 000000007368: BFA5004D  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v41 offset:2048                       // 00000000736C: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 000000007374: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0xe00, v41                            // 00000000737C: 4A1252FF 00000E00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000007384: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007390: BF89FC07 
	ds_store_b16 v41, v10 offset:3584                          // 000000007394: D87C0E00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000739C: 8B7E007E 
	s_cbranch_execz 63                                         // 0000000073A0: BFA5003F  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v8 offset:64                          // 0000000073A4: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000073AC: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000073B8: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 0000000073BC: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000073C4: 8B7E007E 
	s_cbranch_execz 53                                         // 0000000073C8: BFA50035  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v8 offset:128                         // 0000000073CC: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000073D4: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000073E0: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 0000000073E4: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000073EC: 8B7E007E 
	s_cbranch_execz 43                                         // 0000000073F0: BFA5002B  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v8 offset:192                         // 0000000073F4: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 0000000073FC: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007408: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 00000000740C: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007414: 8B7E007E 
	s_cbranch_execz 33                                         // 000000007418: BFA50021  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v8 offset:256                         // 00000000741C: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000007424: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007430: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 000000007434: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000743C: 8B7E007E 
	s_cbranch_execz 23                                         // 000000007440: BFA50017  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000007444: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000744C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007454: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 000000007458: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007460: 8B7E007E 
	s_cbranch_execz 14                                         // 000000007464: BFA5000E  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000007468: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000007470: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007478: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 00000000747C: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007484: 8B7E007E 
	s_cbranch_execz 5                                          // 000000007488: BFA50005  <_amdgpu_cs_main+0x74A0>
	ds_load_u16_d16 v8, v8 offset:448                          // 00000000748C: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007494: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 000000007498: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000074A0: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000074A4: BF89FC07 
	ds_store_b16_d16_hi v32, v81                               // 0000000074A8: DA840000 00005120 
	ds_store_b16_d16_hi v32, v82 offset:64                     // 0000000074B0: DA840040 00005220 
	ds_store_b16_d16_hi v32, v83 offset:128                    // 0000000074B8: DA840080 00005320 
	ds_store_b16_d16_hi v32, v84 offset:192                    // 0000000074C0: DA8400C0 00005420 
	ds_store_b16_d16_hi v32, v85 offset:256                    // 0000000074C8: DA840100 00005520 
	ds_store_b16_d16_hi v32, v86 offset:320                    // 0000000074D0: DA840140 00005620 
	ds_store_b16_d16_hi v32, v87 offset:384                    // 0000000074D8: DA840180 00005720 
	ds_store_b16_d16_hi v32, v88 offset:448                    // 0000000074E0: DA8401C0 00005820 
	s_waitcnt lgkmcnt(0)                                       // 0000000074E8: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000074EC: BE83206A 
	s_cbranch_execz 108                                        // 0000000074F0: BFA5006C  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v41 offset:3584                       // 0000000074F4: DA980E00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 0000000074FC: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 000000007500: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0xe00, v41                            // 000000007508: 4A1052FF 00000E00 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 000000007510: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000007518: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007524: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007528: 6414158A 
	ds_store_b16 v41, v10 offset:3584                          // 00000000752C: D87C0E00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007534: 8B7E007E 
	s_cbranch_execz 90                                         // 000000007538: BFA5005A  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:64                          // 00000000753C: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007544: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000007548: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000007550: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000755C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007560: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000007564: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000756C: 8B7E007E 
	s_cbranch_execz 76                                         // 000000007570: BFA5004C  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007574: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 00000000757C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000007580: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007588: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007594: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007598: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 00000000759C: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000075A4: 8B7E007E 
	s_cbranch_execz 62                                         // 0000000075A8: BFA5003E  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:192                         // 0000000075AC: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000075B4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 0000000075B8: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 0000000075C0: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 0000000075CC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000075D0: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 0000000075D4: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000075DC: 8B7E007E 
	s_cbranch_execz 48                                         // 0000000075E0: BFA50030  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:256                         // 0000000075E4: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000075EC: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 0000000075F0: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000075F8: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007604: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007608: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 00000000760C: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007614: 8B7E007E 
	s_cbranch_execz 34                                         // 000000007618: BFA50022  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:320                         // 00000000761C: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007624: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000007628: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000007630: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007638: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000763C: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000007640: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007648: 8B7E007E 
	s_cbranch_execz 21                                         // 00000000764C: BFA50015  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000007650: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007658: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 00000000765C: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000007664: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000766C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007670: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000007674: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000767C: 8B7E007E 
	s_cbranch_execz 8                                          // 000000007680: BFA50008  <_amdgpu_cs_main+0x76A4>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000007684: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 00000000768C: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000007694: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000007698: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 00000000769C: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000076A4: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000076A8: BF89FC07 
	ds_store_b16_d16_hi v32, v89                               // 0000000076AC: DA840000 00005920 
	ds_store_b16_d16_hi v32, v90 offset:64                     // 0000000076B4: DA840040 00005A20 
	ds_store_b16_d16_hi v32, v91 offset:128                    // 0000000076BC: DA840080 00005B20 
	ds_store_b16_d16_hi v32, v92 offset:192                    // 0000000076C4: DA8400C0 00005C20 
	ds_store_b16_d16_hi v32, v93 offset:256                    // 0000000076CC: DA840100 00005D20 
	ds_store_b16_d16_hi v32, v94 offset:320                    // 0000000076D4: DA840140 00005E20 
	ds_store_b16_d16_hi v32, v95 offset:384                    // 0000000076DC: DA840180 00005F20 
	ds_store_b16_d16_hi v32, v96 offset:448                    // 0000000076E4: DA8401C0 00006020 
	s_waitcnt lgkmcnt(0)                                       // 0000000076EC: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000076F0: BE83206A 
	s_cbranch_execz 77                                         // 0000000076F4: BFA5004D  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v41 offset:2048                       // 0000000076F8: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 000000007700: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0x1000, v41                           // 000000007708: 4A1252FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000007710: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 00000000771C: BF89FC07 
	ds_store_b16 v41, v10 offset:4096                          // 000000007720: D87C1000 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007728: 8B7E007E 
	s_cbranch_execz 63                                         // 00000000772C: BFA5003F  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000007730: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000007738: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007744: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 000000007748: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007750: 8B7E007E 
	s_cbranch_execz 53                                         // 000000007754: BFA50035  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007758: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007760: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000776C: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 000000007770: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007778: 8B7E007E 
	s_cbranch_execz 43                                         // 00000000777C: BFA5002B  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007780: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000007788: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007794: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 000000007798: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000077A0: 8B7E007E 
	s_cbranch_execz 33                                         // 0000000077A4: BFA50021  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v8 offset:256                         // 0000000077A8: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000077B0: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000077BC: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 0000000077C0: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000077C8: 8B7E007E 
	s_cbranch_execz 23                                         // 0000000077CC: BFA50017  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v8 offset:320                         // 0000000077D0: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000077D8: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000077E0: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 0000000077E4: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000077EC: 8B7E007E 
	s_cbranch_execz 14                                         // 0000000077F0: BFA5000E  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v10, v8 offset:384                         // 0000000077F4: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000077FC: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007804: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 000000007808: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007810: 8B7E007E 
	s_cbranch_execz 5                                          // 000000007814: BFA50005  <_amdgpu_cs_main+0x782C>
	ds_load_u16_d16 v8, v8 offset:448                          // 000000007818: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007820: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 000000007824: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000782C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000007830: BF89FC07 
	ds_store_b16_d16_hi v32, v97                               // 000000007834: DA840000 00006120 
	ds_store_b16_d16_hi v32, v98 offset:64                     // 00000000783C: DA840040 00006220 
	ds_store_b16_d16_hi v32, v99 offset:128                    // 000000007844: DA840080 00006320 
	ds_store_b16_d16_hi v32, v100 offset:192                   // 00000000784C: DA8400C0 00006420 
	ds_store_b16_d16_hi v32, v101 offset:256                   // 000000007854: DA840100 00006520 
	ds_store_b16_d16_hi v32, v102 offset:320                   // 00000000785C: DA840140 00006620 
	ds_store_b16_d16_hi v32, v103 offset:384                   // 000000007864: DA840180 00006720 
	ds_store_b16_d16_hi v32, v104 offset:448                   // 00000000786C: DA8401C0 00006820 
	s_waitcnt lgkmcnt(0)                                       // 000000007874: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000007878: BE83206A 
	s_cbranch_execz 108                                        // 00000000787C: BFA5006C  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v41 offset:4096                       // 000000007880: DA981000 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000007888: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 00000000788C: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0x1000, v41                           // 000000007894: 4A1052FF 00001000 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 00000000789C: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000078A4: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000078B0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000078B4: 6414158A 
	ds_store_b16 v41, v10 offset:4096                          // 0000000078B8: D87C1000 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000078C0: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000078C4: BFA5005A  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:64                          // 0000000078C8: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000078D0: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 0000000078D4: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000078DC: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000078E8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000078EC: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 0000000078F0: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000078F8: 8B7E007E 
	s_cbranch_execz 76                                         // 0000000078FC: BFA5004C  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007900: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007908: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 00000000790C: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007914: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007920: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007924: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000007928: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007930: 8B7E007E 
	s_cbranch_execz 62                                         // 000000007934: BFA5003E  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007938: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007940: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000007944: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000794C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007958: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000795C: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000007960: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007968: 8B7E007E 
	s_cbranch_execz 48                                         // 00000000796C: BFA50030  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000007970: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007978: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 00000000797C: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000007984: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007990: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007994: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000007998: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000079A0: 8B7E007E 
	s_cbranch_execz 34                                         // 0000000079A4: BFA50022  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:320                         // 0000000079A8: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000079B0: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 0000000079B4: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000079BC: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000079C4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000079C8: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 0000000079CC: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000079D4: 8B7E007E 
	s_cbranch_execz 21                                         // 0000000079D8: BFA50015  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:384                         // 0000000079DC: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000079E4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 0000000079E8: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000079F0: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000079F8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000079FC: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000007A00: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007A08: 8B7E007E 
	s_cbranch_execz 8                                          // 000000007A0C: BFA50008  <_amdgpu_cs_main+0x7A30>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000007A10: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000007A18: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000007A20: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000007A24: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000007A28: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007A30: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000007A34: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000007A38: BE83206A 
	s_cbranch_execz 108                                        // 000000007A3C: BFA5006C  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v41 offset:2560                       // 000000007A40: DA980A00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000007A48: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:3072                    // 000000007A4C: DA9C0C00 0A000029 
	v_add_nc_u32_e32 v8, 0xa00, v41                            // 000000007A54: 4A1052FF 00000A00 
	v_add_nc_u32_e32 v9, 0xc00, v41                            // 000000007A5C: 4A1252FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000007A64: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007A70: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007A74: 6414158A 
	ds_store_b16 v41, v10 offset:2560                          // 000000007A78: D87C0A00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007A80: 8B7E007E 
	s_cbranch_execz 90                                         // 000000007A84: BFA5005A  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000007A88: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007A90: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000007A94: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000007A9C: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007AA8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007AAC: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000007AB0: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007AB8: 8B7E007E 
	s_cbranch_execz 76                                         // 000000007ABC: BFA5004C  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007AC0: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007AC8: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000007ACC: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007AD4: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007AE0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007AE4: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000007AE8: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007AF0: 8B7E007E 
	s_cbranch_execz 62                                         // 000000007AF4: BFA5003E  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007AF8: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007B00: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000007B04: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000007B0C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007B18: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007B1C: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000007B20: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007B28: 8B7E007E 
	s_cbranch_execz 48                                         // 000000007B2C: BFA50030  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000007B30: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007B38: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000007B3C: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000007B44: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007B50: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007B54: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000007B58: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007B60: 8B7E007E 
	s_cbranch_execz 34                                         // 000000007B64: BFA50022  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000007B68: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007B70: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000007B74: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000007B7C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007B84: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007B88: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000007B8C: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007B94: 8B7E007E 
	s_cbranch_execz 21                                         // 000000007B98: BFA50015  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000007B9C: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007BA4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000007BA8: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000007BB0: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007BB8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007BBC: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000007BC0: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007BC8: 8B7E007E 
	s_cbranch_execz 8                                          // 000000007BCC: BFA50008  <_amdgpu_cs_main+0x7BF0>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000007BD0: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000007BD8: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000007BE0: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000007BE4: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000007BE8: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007BF0: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000007BF4: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000007BF8: BE83206A 
	s_cbranch_execz 108                                        // 000000007BFC: BFA5006C  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v41 offset:3584                       // 000000007C00: DA980E00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000007C08: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:4096                    // 000000007C0C: DA9C1000 0A000029 
	v_add_nc_u32_e32 v8, 0xe00, v41                            // 000000007C14: 4A1052FF 00000E00 
	v_add_nc_u32_e32 v9, 0x1000, v41                           // 000000007C1C: 4A1252FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000007C24: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007C30: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007C34: 6414158A 
	ds_store_b16 v41, v10 offset:3584                          // 000000007C38: D87C0E00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007C40: 8B7E007E 
	s_cbranch_execz 90                                         // 000000007C44: BFA5005A  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000007C48: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007C50: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000007C54: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000007C5C: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007C68: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007C6C: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000007C70: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007C78: 8B7E007E 
	s_cbranch_execz 76                                         // 000000007C7C: BFA5004C  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007C80: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007C88: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000007C8C: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007C94: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007CA0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007CA4: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000007CA8: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007CB0: 8B7E007E 
	s_cbranch_execz 62                                         // 000000007CB4: BFA5003E  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007CB8: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007CC0: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000007CC4: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000007CCC: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007CD8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007CDC: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000007CE0: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007CE8: 8B7E007E 
	s_cbranch_execz 48                                         // 000000007CEC: BFA50030  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000007CF0: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007CF8: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000007CFC: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000007D04: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007D10: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007D14: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000007D18: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007D20: 8B7E007E 
	s_cbranch_execz 34                                         // 000000007D24: BFA50022  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000007D28: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007D30: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000007D34: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000007D3C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007D44: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007D48: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000007D4C: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007D54: 8B7E007E 
	s_cbranch_execz 21                                         // 000000007D58: BFA50015  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000007D5C: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000007D64: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000007D68: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000007D70: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007D78: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000007D7C: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000007D80: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007D88: 8B7E007E 
	s_cbranch_execz 8                                          // 000000007D8C: BFA50008  <_amdgpu_cs_main+0x7DB0>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000007D90: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000007D98: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000007DA0: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000007DA4: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000007DA8: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007DB0: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000007DB4: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 000000007DB8: BE84206A 
	s_cbranch_execz 63                                         // 000000007DBC: BFA5003F  <_amdgpu_cs_main+0x7EBC>
	v_add_nc_u32_e32 v8, s11, v34                              // 000000007DC0: 4A10440B 
	v_or3_b32 v10, s10, v128, 16                               // 000000007DC4: D658000A 0243000A 
	v_add_nc_u32_e32 v11, 0xe00, v41                           // 000000007DCC: 4A1652FF 00000E00 
	s_lshl_b32 s5, s6, 3                                       // 000000007DD4: 84058306 
	s_mov_b32 s7, 0                                            // 000000007DD8: BE870080 
	v_mul_lo_u32 v9, s6, v8                                    // 000000007DDC: D72C0009 00021006 
	v_cmp_gt_u32_e64 s0, s2, v10                               // 000000007DE4: D44C0000 00021402 
	v_add_nc_u32_e32 v10, 0xa00, v41                           // 000000007DEC: 4A1452FF 00000A00 
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000007DF4: BF870123 
	v_add3_u32 v12, v9, s10, v128                              // 000000007DF8: D655000C 06001509 
	v_subrev_nc_u32_e32 v9, 32, v129                           // 000000007E00: 4E1302A0 
	v_lshl_add_u32 v12, v12, 2, 64                             // 000000007E04: D646000C 0301050C 
	s_set_inst_prefetch_distance 0x1                           // 000000007E0C: BF840001 
	s_branch 24                                                // 000000007E10: BFA00018  <_amdgpu_cs_main+0x7E74>
	s_nop 0                                                    // 000000007E14: BF800000 
	s_nop 0                                                    // 000000007E18: BF800000 
	s_nop 0                                                    // 000000007E1C: BF800000 
	s_nop 0                                                    // 000000007E20: BF800000 
	s_nop 0                                                    // 000000007E24: BF800000 
	s_nop 0                                                    // 000000007E28: BF800000 
	s_nop 0                                                    // 000000007E2C: BF800000 
	s_nop 0                                                    // 000000007E30: BF800000 
	s_nop 0                                                    // 000000007E34: BF800000 
	s_nop 0                                                    // 000000007E38: BF800000 
	s_nop 0                                                    // 000000007E3C: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007E40: 8C7E037E 
	v_add_nc_u32_e32 v9, 32, v9                                // 000000007E44: 4A1212A0 
	v_add_nc_u32_e32 v10, 64, v10                              // 000000007E48: 4A1414C0 
	v_add_nc_u32_e32 v11, 64, v11                              // 000000007E4C: 4A1616C0 
	v_add_nc_u32_e32 v12, s5, v12                              // 000000007E50: 4A181805 
	v_add_nc_u32_e32 v8, 2, v8                                 // 000000007E54: 4A101082 
	v_cmp_lt_u32_e64 s3, 0xdf, v9                              // 000000007E58: D4490003 000212FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000007E64: 8C070703 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007E68: BF870009 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 000000007E6C: 917E077E 
	s_cbranch_execz 18                                         // 000000007E70: BFA50012  <_amdgpu_cs_main+0x7EBC>
	v_cmp_gt_u32_e64 s3, s1, v8                                // 000000007E74: D44C0003 00021001 
	s_and_b32 s8, s3, s0                                       // 000000007E7C: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007E80: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000007E84: BE832008 
	s_cbranch_execz 65517                                      // 000000007E88: BFA5FFED  <_amdgpu_cs_main+0x7E40>
	ds_load_u16_d16 v13, v10                                   // 000000007E8C: DA980000 0D00000A 
	s_waitcnt lgkmcnt(0)                                       // 000000007E94: BF89FC07 
	ds_load_u16_d16_hi v13, v11                                // 000000007E98: DA9C0000 0D00000B 
	s_waitcnt lgkmcnt(0)                                       // 000000007EA0: BF89FC07 
	v_add_f16_e32 v13, v141/*Invalid register, operand has 'VS_32_Lo128' register class*/, v13// 000000007EA4: 641A1B8D 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007EA8: BF870001 
	v_cvt_f32_f16_e32 v13, v13                                 // 000000007EAC: 7E1A170D 
	buffer_store_b32 v13, v12, s[20:23], 0 offen               // 000000007EB0: E0680000 80450D0C 
	s_branch 65505                                             // 000000007EB8: BFA0FFE1  <_amdgpu_cs_main+0x7E40>
	s_set_inst_prefetch_distance 0x2                           // 000000007EBC: BF840002 
	s_or_b32 exec_lo, exec_lo, s4                              // 000000007EC0: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 000000007EC4: BC7C0000 
	s_clause 0x1                                               // 000000007EC8: BF850001 
	scratch_load_b128 v[0:3], off, off                         // 000000007ECC: DC5D0000 007C0000 
	scratch_load_b128 v[4:7], off, off offset:16               // 000000007ED4: DC5D0010 047C0000 
	s_waitcnt vmcnt(1)                                         // 000000007EDC: BF8907F7 
	ds_store_b16 v32, v0                                       // 000000007EE0: D87C0000 00000020 
	ds_store_b16 v32, v1 offset:64                             // 000000007EE8: D87C0040 00000120 
	ds_store_b16 v32, v2 offset:128                            // 000000007EF0: D87C0080 00000220 
	ds_store_b16 v32, v3 offset:192                            // 000000007EF8: D87C00C0 00000320 
	s_waitcnt vmcnt(0)                                         // 000000007F00: BF8903F7 
	ds_store_b16 v32, v4 offset:256                            // 000000007F04: D87C0100 00000420 
	ds_store_b16 v32, v5 offset:320                            // 000000007F0C: D87C0140 00000520 
	ds_store_b16 v32, v6 offset:384                            // 000000007F14: D87C0180 00000620 
	ds_store_b16 v32, v7 offset:448                            // 000000007F1C: D87C01C0 00000720 
	s_waitcnt lgkmcnt(0)                                       // 000000007F24: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000007F28: BE83206A 
	s_cbranch_execz 77                                         // 000000007F2C: BFA5004D  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v41 offset:2048                       // 000000007F30: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 000000007F38: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0xa00, v41                            // 000000007F40: 4A1252FF 00000A00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000007F48: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000007F54: BF89FC07 
	ds_store_b16 v41, v10 offset:2560                          // 000000007F58: D87C0A00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007F60: 8B7E007E 
	s_cbranch_execz 63                                         // 000000007F64: BFA5003F  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000007F68: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000007F70: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000007F7C: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 000000007F80: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007F88: 8B7E007E 
	s_cbranch_execz 53                                         // 000000007F8C: BFA50035  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000007F90: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000007F98: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000007FA4: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 000000007FA8: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007FB0: 8B7E007E 
	s_cbranch_execz 43                                         // 000000007FB4: BFA5002B  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000007FB8: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000007FC0: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000007FCC: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 000000007FD0: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000007FD8: 8B7E007E 
	s_cbranch_execz 33                                         // 000000007FDC: BFA50021  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000007FE0: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000007FE8: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000007FF4: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 000000007FF8: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008000: 8B7E007E 
	s_cbranch_execz 23                                         // 000000008004: BFA50017  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000008008: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000008010: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008018: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 00000000801C: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008024: 8B7E007E 
	s_cbranch_execz 14                                         // 000000008028: BFA5000E  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v10, v8 offset:384                         // 00000000802C: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000008034: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000803C: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 000000008040: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008048: 8B7E007E 
	s_cbranch_execz 5                                          // 00000000804C: BFA50005  <_amdgpu_cs_main+0x8064>
	ds_load_u16_d16 v8, v8 offset:448                          // 000000008050: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008058: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 00000000805C: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008064: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000008068: BF89FC07 
	ds_store_b16 v32, v208                                     // 00000000806C: D87C0000 0000D020 
	ds_store_b16 v32, v209 offset:64                           // 000000008074: D87C0040 0000D120 
	ds_store_b16 v32, v210 offset:128                          // 00000000807C: D87C0080 0000D220 
	ds_store_b16 v32, v211 offset:192                          // 000000008084: D87C00C0 0000D320 
	ds_store_b16 v32, v212 offset:256                          // 00000000808C: D87C0100 0000D420 
	ds_store_b16 v32, v213 offset:320                          // 000000008094: D87C0140 0000D520 
	ds_store_b16 v32, v214 offset:384                          // 00000000809C: D87C0180 0000D620 
	ds_store_b16 v32, v215 offset:448                          // 0000000080A4: D87C01C0 0000D720 
	s_waitcnt lgkmcnt(0)                                       // 0000000080AC: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000080B0: BE83206A 
	s_cbranch_execz 108                                        // 0000000080B4: BFA5006C  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v41 offset:2560                       // 0000000080B8: DA980A00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 0000000080C0: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 0000000080C4: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0xa00, v41                            // 0000000080CC: 4A1052FF 00000A00 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 0000000080D4: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000080DC: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000080E8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000080EC: 6414158A 
	ds_store_b16 v41, v10 offset:2560                          // 0000000080F0: D87C0A00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000080F8: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000080FC: BFA5005A  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008100: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008108: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 00000000810C: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000008114: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008120: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008124: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000008128: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008130: 8B7E007E 
	s_cbranch_execz 76                                         // 000000008134: BFA5004C  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000008138: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008140: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000008144: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 00000000814C: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008158: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000815C: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000008160: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008168: 8B7E007E 
	s_cbranch_execz 62                                         // 00000000816C: BFA5003E  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008170: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008178: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 00000000817C: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000008184: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000008190: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008194: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000008198: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000081A0: 8B7E007E 
	s_cbranch_execz 48                                         // 0000000081A4: BFA50030  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:256                         // 0000000081A8: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000081B0: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 0000000081B4: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000081BC: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000081C8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000081CC: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 0000000081D0: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000081D8: 8B7E007E 
	s_cbranch_execz 34                                         // 0000000081DC: BFA50022  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:320                         // 0000000081E0: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000081E8: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 0000000081EC: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000081F4: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000081FC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008200: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000008204: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000820C: 8B7E007E 
	s_cbranch_execz 21                                         // 000000008210: BFA50015  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000008214: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 00000000821C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000008220: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000008228: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008230: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008234: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000008238: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008240: 8B7E007E 
	s_cbranch_execz 8                                          // 000000008244: BFA50008  <_amdgpu_cs_main+0x8268>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000008248: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000008250: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000008258: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 00000000825C: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000008260: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008268: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 00000000826C: BF89FC07 
	ds_store_b16 v32, v50                                      // 000000008270: D87C0000 00003220 
	ds_store_b16 v32, v51 offset:64                            // 000000008278: D87C0040 00003320 
	ds_store_b16 v32, v52 offset:128                           // 000000008280: D87C0080 00003420 
	ds_store_b16 v32, v53 offset:192                           // 000000008288: D87C00C0 00003520 
	ds_store_b16 v32, v54 offset:256                           // 000000008290: D87C0100 00003620 
	ds_store_b16 v32, v55 offset:320                           // 000000008298: D87C0140 00003720 
	ds_store_b16 v32, v56 offset:384                           // 0000000082A0: D87C0180 00003820 
	ds_store_b16 v32, v57 offset:448                           // 0000000082A8: D87C01C0 00003920 
	s_waitcnt lgkmcnt(0)                                       // 0000000082B0: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000082B4: BE83206A 
	s_cbranch_execz 77                                         // 0000000082B8: BFA5004D  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v41 offset:2048                       // 0000000082BC: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 0000000082C4: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0xc00, v41                            // 0000000082CC: 4A1252FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000082D4: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000082E0: BF89FC07 
	ds_store_b16 v41, v10 offset:3072                          // 0000000082E4: D87C0C00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000082EC: 8B7E007E 
	s_cbranch_execz 63                                         // 0000000082F0: BFA5003F  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v8 offset:64                          // 0000000082F4: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000082FC: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008308: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 00000000830C: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008314: 8B7E007E 
	s_cbranch_execz 53                                         // 000000008318: BFA50035  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v8 offset:128                         // 00000000831C: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000008324: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008330: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 000000008334: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000833C: 8B7E007E 
	s_cbranch_execz 43                                         // 000000008340: BFA5002B  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008344: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000834C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000008358: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 00000000835C: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008364: 8B7E007E 
	s_cbranch_execz 33                                         // 000000008368: BFA50021  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v8 offset:256                         // 00000000836C: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008374: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000008380: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 000000008384: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000838C: 8B7E007E 
	s_cbranch_execz 23                                         // 000000008390: BFA50017  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000008394: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000839C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000083A4: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 0000000083A8: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000083B0: 8B7E007E 
	s_cbranch_execz 14                                         // 0000000083B4: BFA5000E  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v10, v8 offset:384                         // 0000000083B8: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000083C0: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000083C8: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 0000000083CC: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000083D4: 8B7E007E 
	s_cbranch_execz 5                                          // 0000000083D8: BFA50005  <_amdgpu_cs_main+0x83F0>
	ds_load_u16_d16 v8, v8 offset:448                          // 0000000083DC: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000083E4: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 0000000083E8: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000083F0: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000083F4: BF89FC07 
	ds_store_b16 v32, v58                                      // 0000000083F8: D87C0000 00003A20 
	ds_store_b16 v32, v59 offset:64                            // 000000008400: D87C0040 00003B20 
	ds_store_b16 v32, v60 offset:128                           // 000000008408: D87C0080 00003C20 
	ds_store_b16 v32, v61 offset:192                           // 000000008410: D87C00C0 00003D20 
	ds_store_b16 v32, v62 offset:256                           // 000000008418: D87C0100 00003E20 
	ds_store_b16 v32, v63 offset:320                           // 000000008420: D87C0140 00003F20 
	ds_store_b16 v32, v64 offset:384                           // 000000008428: D87C0180 00004020 
	ds_store_b16 v32, v65 offset:448                           // 000000008430: D87C01C0 00004120 
	s_waitcnt lgkmcnt(0)                                       // 000000008438: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 00000000843C: BE83206A 
	s_cbranch_execz 108                                        // 000000008440: BFA5006C  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v41 offset:3072                       // 000000008444: DA980C00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 00000000844C: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 000000008450: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0xc00, v41                            // 000000008458: 4A1052FF 00000C00 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 000000008460: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000008468: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000008474: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008478: 6414158A 
	ds_store_b16 v41, v10 offset:3072                          // 00000000847C: D87C0C00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008484: 8B7E007E 
	s_cbranch_execz 90                                         // 000000008488: BFA5005A  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:64                          // 00000000848C: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008494: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000008498: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000084A0: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000084AC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000084B0: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 0000000084B4: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000084BC: 8B7E007E 
	s_cbranch_execz 76                                         // 0000000084C0: BFA5004C  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:128                         // 0000000084C4: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000084CC: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 0000000084D0: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000084D8: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000084E4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000084E8: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 0000000084EC: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000084F4: 8B7E007E 
	s_cbranch_execz 62                                         // 0000000084F8: BFA5003E  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:192                         // 0000000084FC: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008504: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000008508: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000008510: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 00000000851C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008520: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000008524: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000852C: 8B7E007E 
	s_cbranch_execz 48                                         // 000000008530: BFA50030  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000008534: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 00000000853C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000008540: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008548: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000008554: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008558: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 00000000855C: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008564: 8B7E007E 
	s_cbranch_execz 34                                         // 000000008568: BFA50022  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:320                         // 00000000856C: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008574: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000008578: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000008580: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008588: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000858C: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000008590: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008598: 8B7E007E 
	s_cbranch_execz 21                                         // 00000000859C: BFA50015  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:384                         // 0000000085A0: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000085A8: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 0000000085AC: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000085B4: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000085BC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000085C0: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 0000000085C4: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000085CC: 8B7E007E 
	s_cbranch_execz 8                                          // 0000000085D0: BFA50008  <_amdgpu_cs_main+0x85F4>
	ds_load_u16_d16 v10, v8 offset:448                         // 0000000085D4: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 0000000085DC: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 0000000085E4: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 0000000085E8: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 0000000085EC: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000085F4: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000085F8: BF89FC07 
	ds_store_b16 v32, v105                                     // 0000000085FC: D87C0000 00006920 
	ds_store_b16 v32, v106 offset:64                           // 000000008604: D87C0040 00006A20 
	ds_store_b16 v32, v107 offset:128                          // 00000000860C: D87C0080 00006B20 
	ds_store_b16 v32, v108 offset:192                          // 000000008614: D87C00C0 00006C20 
	ds_store_b16 v32, v109 offset:256                          // 00000000861C: D87C0100 00006D20 
	ds_store_b16 v32, v110 offset:320                          // 000000008624: D87C0140 00006E20 
	ds_store_b16 v32, v111 offset:384                          // 00000000862C: D87C0180 00006F20 
	ds_store_b16 v32, v112 offset:448                          // 000000008634: D87C01C0 00007020 
	s_waitcnt lgkmcnt(0)                                       // 00000000863C: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000008640: BE83206A 
	s_cbranch_execz 77                                         // 000000008644: BFA5004D  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v41 offset:2048                       // 000000008648: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 000000008650: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0xe00, v41                            // 000000008658: 4A1252FF 00000E00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000008660: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 00000000866C: BF89FC07 
	ds_store_b16 v41, v10 offset:3584                          // 000000008670: D87C0E00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008678: 8B7E007E 
	s_cbranch_execz 63                                         // 00000000867C: BFA5003F  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008680: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000008688: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008694: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 000000008698: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000086A0: 8B7E007E 
	s_cbranch_execz 53                                         // 0000000086A4: BFA50035  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v8 offset:128                         // 0000000086A8: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000086B0: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000086BC: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 0000000086C0: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000086C8: 8B7E007E 
	s_cbranch_execz 43                                         // 0000000086CC: BFA5002B  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v8 offset:192                         // 0000000086D0: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 0000000086D8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 0000000086E4: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 0000000086E8: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000086F0: 8B7E007E 
	s_cbranch_execz 33                                         // 0000000086F4: BFA50021  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v8 offset:256                         // 0000000086F8: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008700: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 00000000870C: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 000000008710: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008718: 8B7E007E 
	s_cbranch_execz 23                                         // 00000000871C: BFA50017  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000008720: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000008728: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008730: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 000000008734: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000873C: 8B7E007E 
	s_cbranch_execz 14                                         // 000000008740: BFA5000E  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000008744: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 00000000874C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008754: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 000000008758: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008760: 8B7E007E 
	s_cbranch_execz 5                                          // 000000008764: BFA50005  <_amdgpu_cs_main+0x877C>
	ds_load_u16_d16 v8, v8 offset:448                          // 000000008768: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008770: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 000000008774: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000877C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000008780: BF89FC07 
	ds_store_b16 v32, v113                                     // 000000008784: D87C0000 00007120 
	ds_store_b16 v32, v114 offset:64                           // 00000000878C: D87C0040 00007220 
	ds_store_b16 v32, v115 offset:128                          // 000000008794: D87C0080 00007320 
	ds_store_b16 v32, v116 offset:192                          // 00000000879C: D87C00C0 00007420 
	ds_store_b16 v32, v117 offset:256                          // 0000000087A4: D87C0100 00007520 
	ds_store_b16 v32, v118 offset:320                          // 0000000087AC: D87C0140 00007620 
	ds_store_b16 v32, v119 offset:384                          // 0000000087B4: D87C0180 00007720 
	ds_store_b16 v32, v120 offset:448                          // 0000000087BC: D87C01C0 00007820 
	s_waitcnt lgkmcnt(0)                                       // 0000000087C4: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000087C8: BE83206A 
	s_cbranch_execz 108                                        // 0000000087CC: BFA5006C  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v41 offset:3584                       // 0000000087D0: DA980E00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 0000000087D8: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 0000000087DC: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0xe00, v41                            // 0000000087E4: 4A1052FF 00000E00 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 0000000087EC: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000087F4: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000008800: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008804: 6414158A 
	ds_store_b16 v41, v10 offset:3584                          // 000000008808: D87C0E00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008810: 8B7E007E 
	s_cbranch_execz 90                                         // 000000008814: BFA5005A  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008818: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008820: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000008824: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 00000000882C: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008838: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000883C: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000008840: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008848: 8B7E007E 
	s_cbranch_execz 76                                         // 00000000884C: BFA5004C  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000008850: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008858: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 00000000885C: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000008864: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008870: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008874: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000008878: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008880: 8B7E007E 
	s_cbranch_execz 62                                         // 000000008884: BFA5003E  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008888: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008890: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000008894: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000889C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 0000000088A8: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000088AC: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 0000000088B0: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000088B8: 8B7E007E 
	s_cbranch_execz 48                                         // 0000000088BC: BFA50030  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:256                         // 0000000088C0: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 0000000088C8: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 0000000088CC: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000088D4: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000088E0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 0000000088E4: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 0000000088E8: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000088F0: 8B7E007E 
	s_cbranch_execz 34                                         // 0000000088F4: BFA50022  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:320                         // 0000000088F8: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008900: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000008904: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000890C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008914: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008918: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 00000000891C: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008924: 8B7E007E 
	s_cbranch_execz 21                                         // 000000008928: BFA50015  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:384                         // 00000000892C: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008934: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000008938: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000008940: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008948: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 00000000894C: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000008950: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008958: 8B7E007E 
	s_cbranch_execz 8                                          // 00000000895C: BFA50008  <_amdgpu_cs_main+0x8980>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000008960: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000008968: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000008970: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000008974: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000008978: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008980: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000008984: BF89FC07 
	ds_store_b16 v32, v168                                     // 000000008988: D87C0000 0000A820 
	ds_store_b16 v32, v169 offset:64                           // 000000008990: D87C0040 0000A920 
	ds_store_b16 v32, v170 offset:128                          // 000000008998: D87C0080 0000AA20 
	ds_store_b16 v32, v171 offset:192                          // 0000000089A0: D87C00C0 0000AB20 
	ds_store_b16 v32, v172 offset:256                          // 0000000089A8: D87C0100 0000AC20 
	ds_store_b16 v32, v173 offset:320                          // 0000000089B0: D87C0140 0000AD20 
	ds_store_b16 v32, v174 offset:384                          // 0000000089B8: D87C0180 0000AE20 
	ds_store_b16 v32, v175 offset:448                          // 0000000089C0: D87C01C0 0000AF20 
	s_waitcnt lgkmcnt(0)                                       // 0000000089C8: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000089CC: BE83206A 
	s_cbranch_execz 77                                         // 0000000089D0: BFA5004D  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v41 offset:2048                       // 0000000089D4: DA980800 0A000029 
	v_add_nc_u32_e32 v8, 0x800, v41                            // 0000000089DC: 4A1052FF 00000800 
	v_add_nc_u32_e32 v9, 0x1000, v41                           // 0000000089E4: 4A1252FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 0000000089EC: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000089F8: BF89FC07 
	ds_store_b16 v41, v10 offset:4096                          // 0000000089FC: D87C1000 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008A04: 8B7E007E 
	s_cbranch_execz 63                                         // 000000008A08: BFA5003F  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008A0C: DA980040 0A000008 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000008A14: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008A20: BF89FC07 
	ds_store_b16 v9, v10 offset:64                             // 000000008A24: D87C0040 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008A2C: 8B7E007E 
	s_cbranch_execz 53                                         // 000000008A30: BFA50035  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000008A34: DA980080 0A000008 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000008A3C: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008A48: BF89FC07 
	ds_store_b16 v9, v10 offset:128                            // 000000008A4C: D87C0080 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008A54: 8B7E007E 
	s_cbranch_execz 43                                         // 000000008A58: BFA5002B  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008A5C: DA9800C0 0A000008 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000008A64: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000008A70: BF89FC07 
	ds_store_b16 v9, v10 offset:192                            // 000000008A74: D87C00C0 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008A7C: 8B7E007E 
	s_cbranch_execz 33                                         // 000000008A80: BFA50021  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000008A84: DA980100 0A000008 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008A8C: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000008A98: BF89FC07 
	ds_store_b16 v9, v10 offset:256                            // 000000008A9C: D87C0100 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008AA4: 8B7E007E 
	s_cbranch_execz 23                                         // 000000008AA8: BFA50017  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000008AAC: DA980140 0A000008 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000008AB4: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008ABC: BF89FC07 
	ds_store_b16 v9, v10 offset:320                            // 000000008AC0: D87C0140 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008AC8: 8B7E007E 
	s_cbranch_execz 14                                         // 000000008ACC: BFA5000E  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000008AD0: DA980180 0A000008 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000008AD8: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008AE0: BF89FC07 
	ds_store_b16 v9, v10 offset:384                            // 000000008AE4: D87C0180 00000A09 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008AEC: 8B7E007E 
	s_cbranch_execz 5                                          // 000000008AF0: BFA50005  <_amdgpu_cs_main+0x8B08>
	ds_load_u16_d16 v8, v8 offset:448                          // 000000008AF4: DA9801C0 08000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008AFC: BF89FC07 
	ds_store_b16 v9, v8 offset:448                             // 000000008B00: D87C01C0 00000809 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008B08: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000008B0C: BF89FC07 
	ds_store_b16 v32, v160                                     // 000000008B10: D87C0000 0000A020 
	ds_store_b16 v32, v161 offset:64                           // 000000008B18: D87C0040 0000A120 
	ds_store_b16 v32, v162 offset:128                          // 000000008B20: D87C0080 0000A220 
	ds_store_b16 v32, v163 offset:192                          // 000000008B28: D87C00C0 0000A320 
	ds_store_b16 v32, v164 offset:256                          // 000000008B30: D87C0100 0000A420 
	ds_store_b16 v32, v165 offset:320                          // 000000008B38: D87C0140 0000A520 
	ds_store_b16 v32, v166 offset:384                          // 000000008B40: D87C0180 0000A620 
	ds_store_b16 v32, v167 offset:448                          // 000000008B48: D87C01C0 0000A720 
	s_waitcnt lgkmcnt(0)                                       // 000000008B50: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000008B54: BE83206A 
	s_cbranch_execz 108                                        // 000000008B58: BFA5006C  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v41 offset:4096                       // 000000008B5C: DA981000 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000008B64: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:2048                    // 000000008B68: DA9C0800 0A000029 
	v_add_nc_u32_e32 v8, 0x1000, v41                           // 000000008B70: 4A1052FF 00001000 
	v_add_nc_u32_e32 v9, 0x800, v41                            // 000000008B78: 4A1252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000008B80: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000008B8C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008B90: 6414158A 
	ds_store_b16 v41, v10 offset:4096                          // 000000008B94: D87C1000 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008B9C: 8B7E007E 
	s_cbranch_execz 90                                         // 000000008BA0: BFA5005A  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008BA4: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008BAC: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000008BB0: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000008BB8: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008BC4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008BC8: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000008BCC: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008BD4: 8B7E007E 
	s_cbranch_execz 76                                         // 000000008BD8: BFA5004C  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000008BDC: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008BE4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000008BE8: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000008BF0: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008BFC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008C00: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000008C04: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008C0C: 8B7E007E 
	s_cbranch_execz 62                                         // 000000008C10: BFA5003E  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008C14: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008C1C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000008C20: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000008C28: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000008C34: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008C38: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000008C3C: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008C44: 8B7E007E 
	s_cbranch_execz 48                                         // 000000008C48: BFA50030  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000008C4C: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008C54: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000008C58: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008C60: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000008C6C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008C70: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000008C74: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008C7C: 8B7E007E 
	s_cbranch_execz 34                                         // 000000008C80: BFA50022  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000008C84: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008C8C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000008C90: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000008C98: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008CA0: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008CA4: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000008CA8: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008CB0: 8B7E007E 
	s_cbranch_execz 21                                         // 000000008CB4: BFA50015  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000008CB8: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008CC0: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000008CC4: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000008CCC: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008CD4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008CD8: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000008CDC: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008CE4: 8B7E007E 
	s_cbranch_execz 8                                          // 000000008CE8: BFA50008  <_amdgpu_cs_main+0x8D0C>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000008CEC: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000008CF4: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000008CFC: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000008D00: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000008D04: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008D0C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000008D10: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000008D14: BE83206A 
	s_cbranch_execz 108                                        // 000000008D18: BFA5006C  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v41 offset:2560                       // 000000008D1C: DA980A00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000008D24: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:3072                    // 000000008D28: DA9C0C00 0A000029 
	v_add_nc_u32_e32 v8, 0xa00, v41                            // 000000008D30: 4A1052FF 00000A00 
	v_add_nc_u32_e32 v9, 0xc00, v41                            // 000000008D38: 4A1252FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000008D40: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000008D4C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008D50: 6414158A 
	ds_store_b16 v41, v10 offset:2560                          // 000000008D54: D87C0A00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008D5C: 8B7E007E 
	s_cbranch_execz 90                                         // 000000008D60: BFA5005A  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008D64: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008D6C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000008D70: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000008D78: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008D84: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008D88: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000008D8C: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008D94: 8B7E007E 
	s_cbranch_execz 76                                         // 000000008D98: BFA5004C  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000008D9C: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008DA4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000008DA8: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000008DB0: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008DBC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008DC0: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000008DC4: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008DCC: 8B7E007E 
	s_cbranch_execz 62                                         // 000000008DD0: BFA5003E  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008DD4: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008DDC: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000008DE0: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000008DE8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000008DF4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008DF8: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000008DFC: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008E04: 8B7E007E 
	s_cbranch_execz 48                                         // 000000008E08: BFA50030  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000008E0C: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008E14: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000008E18: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008E20: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000008E2C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008E30: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000008E34: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008E3C: 8B7E007E 
	s_cbranch_execz 34                                         // 000000008E40: BFA50022  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000008E44: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008E4C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000008E50: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000008E58: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008E60: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008E64: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000008E68: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008E70: 8B7E007E 
	s_cbranch_execz 21                                         // 000000008E74: BFA50015  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000008E78: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008E80: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000008E84: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000008E8C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008E94: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008E98: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 000000008E9C: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008EA4: 8B7E007E 
	s_cbranch_execz 8                                          // 000000008EA8: BFA50008  <_amdgpu_cs_main+0x8ECC>
	ds_load_u16_d16 v10, v8 offset:448                         // 000000008EAC: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000008EB4: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 000000008EBC: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000008EC0: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000008EC4: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008ECC: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000008ED0: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000008ED4: BE83206A 
	s_cbranch_execz 108                                        // 000000008ED8: BFA5006C  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v41 offset:3584                       // 000000008EDC: DA980E00 0A000029 
	s_waitcnt lgkmcnt(0)                                       // 000000008EE4: BF89FC07 
	ds_load_u16_d16_hi v10, v41 offset:4096                    // 000000008EE8: DA9C1000 0A000029 
	v_add_nc_u32_e32 v8, 0xe00, v41                            // 000000008EF0: 4A1052FF 00000E00 
	v_add_nc_u32_e32 v9, 0x1000, v41                           // 000000008EF8: 4A1252FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000008F00: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000008F0C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008F10: 6414158A 
	ds_store_b16 v41, v10 offset:3584                          // 000000008F14: D87C0E00 00000A29 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008F1C: 8B7E007E 
	s_cbranch_execz 90                                         // 000000008F20: BFA5005A  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:64                          // 000000008F24: DA980040 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008F2C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:64                       // 000000008F30: DA9C0040 0A000009 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000008F38: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000008F44: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008F48: 6414158A 
	ds_store_b16 v8, v10 offset:64                             // 000000008F4C: D87C0040 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008F54: 8B7E007E 
	s_cbranch_execz 76                                         // 000000008F58: BFA5004C  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:128                         // 000000008F5C: DA980080 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008F64: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:128                      // 000000008F68: DA9C0080 0A000009 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000008F70: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000008F7C: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008F80: 6414158A 
	ds_store_b16 v8, v10 offset:128                            // 000000008F84: D87C0080 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008F8C: 8B7E007E 
	s_cbranch_execz 62                                         // 000000008F90: BFA5003E  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:192                         // 000000008F94: DA9800C0 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008F9C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:192                      // 000000008FA0: DA9C00C0 0A000009 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000008FA8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000008FB4: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008FB8: 6414158A 
	ds_store_b16 v8, v10 offset:192                            // 000000008FBC: D87C00C0 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008FC4: 8B7E007E 
	s_cbranch_execz 48                                         // 000000008FC8: BFA50030  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:256                         // 000000008FCC: DA980100 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000008FD4: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:256                      // 000000008FD8: DA9C0100 0A000009 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000008FE0: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000008FEC: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000008FF0: 6414158A 
	ds_store_b16 v8, v10 offset:256                            // 000000008FF4: D87C0100 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000008FFC: 8B7E007E 
	s_cbranch_execz 34                                         // 000000009000: BFA50022  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:320                         // 000000009004: DA980140 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 00000000900C: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:320                      // 000000009010: DA9C0140 0A000009 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000009018: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009020: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000009024: 6414158A 
	ds_store_b16 v8, v10 offset:320                            // 000000009028: D87C0140 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009030: 8B7E007E 
	s_cbranch_execz 21                                         // 000000009034: BFA50015  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:384                         // 000000009038: DA980180 0A000008 
	s_waitcnt lgkmcnt(0)                                       // 000000009040: BF89FC07 
	ds_load_u16_d16_hi v10, v9 offset:384                      // 000000009044: DA9C0180 0A000009 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 00000000904C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009054: BF89FC07 
	v_add_f16_e32 v10, v138/*Invalid register, operand has 'VS_32_Lo128' register class*/, v10// 000000009058: 6414158A 
	ds_store_b16 v8, v10 offset:384                            // 00000000905C: D87C0180 00000A08 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009064: 8B7E007E 
	s_cbranch_execz 8                                          // 000000009068: BFA50008  <_amdgpu_cs_main+0x908C>
	ds_load_u16_d16 v10, v8 offset:448                         // 00000000906C: DA9801C0 0A000008 
	ds_load_u16_d16 v9, v9 offset:448                          // 000000009074: DA9801C0 09000009 
	s_waitcnt lgkmcnt(0)                                       // 00000000907C: BF89FC07 
	v_add_f16_e32 v9, v9, v10                                  // 000000009080: 64121509 
	ds_store_b16 v8, v9 offset:448                             // 000000009084: D87C01C0 00000908 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000908C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000009090: BF89FC07 
	s_and_saveexec_b32 s4, vcc_lo                              // 000000009094: BE84206A 
	s_cbranch_execz 56                                         // 000000009098: BFA50038  <_amdgpu_cs_main+0x917C>
	v_add3_u32 v8, v34, s11, 16                                // 00000000909C: D6550008 02401722 
	v_or_b32_e32 v9, s10, v128                                 // 0000000090A4: 3813000A 
	v_add_nc_u32_e32 v10, 0xa00, v41                           // 0000000090A8: 4A1452FF 00000A00 
	v_add_nc_u32_e32 v11, 0xe00, v41                           // 0000000090B0: 4A1652FF 00000E00 
	s_lshl_b32 s5, s6, 3                                       // 0000000090B8: 84058306 
	v_mad_u64_u32 v[12:13], null, s6, v8, s[10:11]             // 0000000090BC: D6FE7C0C 002A1006 
	v_cmp_gt_u32_e64 s0, s2, v9                                // 0000000090C4: D44C0000 00021202 
	v_subrev_nc_u32_e32 v9, 32, v129                           // 0000000090CC: 4E1302A0 
	s_mov_b32 s7, 0                                            // 0000000090D0: BE870080 
	v_add_lshl_u32 v12, v12, v128, 2                           // 0000000090D4: D647000C 020B010C 
	s_set_inst_prefetch_distance 0x1                           // 0000000090DC: BF840001 
	s_branch 20                                                // 0000000090E0: BFA00014  <_amdgpu_cs_main+0x9134>
	s_nop 0                                                    // 0000000090E4: BF800000 
	s_nop 0                                                    // 0000000090E8: BF800000 
	s_nop 0                                                    // 0000000090EC: BF800000 
	s_nop 0                                                    // 0000000090F0: BF800000 
	s_nop 0                                                    // 0000000090F4: BF800000 
	s_nop 0                                                    // 0000000090F8: BF800000 
	s_nop 0                                                    // 0000000090FC: BF800000 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009100: 8C7E037E 
	v_add_nc_u32_e32 v9, 32, v9                                // 000000009104: 4A1212A0 
	v_add_nc_u32_e32 v10, 64, v10                              // 000000009108: 4A1414C0 
	v_add_nc_u32_e32 v11, 64, v11                              // 00000000910C: 4A1616C0 
	v_add_nc_u32_e32 v8, 2, v8                                 // 000000009110: 4A101082 
	v_add_nc_u32_e32 v12, s5, v12                              // 000000009114: 4A181805 
	v_cmp_lt_u32_e64 s3, 0xdf, v9                              // 000000009118: D4490003 000212FF 000000DF 
	s_or_b32 s7, s3, s7                                        // 000000009124: 8C070703 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000009128: BF870009 
	s_and_not1_b32 exec_lo, exec_lo, s7                        // 00000000912C: 917E077E 
	s_cbranch_execz 18                                         // 000000009130: BFA50012  <_amdgpu_cs_main+0x917C>
	v_cmp_gt_u32_e64 s3, s1, v8                                // 000000009134: D44C0003 00021001 
	s_and_b32 s8, s3, s0                                       // 00000000913C: 8B080003 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000009140: BF870009 
	s_and_saveexec_b32 s3, s8                                  // 000000009144: BE832008 
	s_cbranch_execz 65517                                      // 000000009148: BFA5FFED  <_amdgpu_cs_main+0x9100>
	ds_load_u16_d16 v13, v10                                   // 00000000914C: DA980000 0D00000A 
	s_waitcnt lgkmcnt(0)                                       // 000000009154: BF89FC07 
	ds_load_u16_d16_hi v13, v11                                // 000000009158: DA9C0000 0D00000B 
	s_waitcnt lgkmcnt(0)                                       // 000000009160: BF89FC07 
	v_add_f16_e32 v13, v141/*Invalid register, operand has 'VS_32_Lo128' register class*/, v13// 000000009164: 641A1B8D 
	s_delay_alu instid0(VALU_DEP_1)                            // 000000009168: BF870001 
	v_cvt_f32_f16_e32 v13, v13                                 // 00000000916C: 7E1A170D 
	buffer_store_b32 v13, v12, s[20:23], 0 offen               // 000000009170: E0680000 80450D0C 
	s_branch 65505                                             // 000000009178: BFA0FFE1  <_amdgpu_cs_main+0x9100>
	s_set_inst_prefetch_distance 0x2                           // 00000000917C: BF840002 
	s_or_b32 exec_lo, exec_lo, s4                              // 000000009180: 8C7E047E 
	s_waitcnt_vscnt null, 0x0                                  // 000000009184: BC7C0000 
	s_clause 0x1                                               // 000000009188: BF850001 
	scratch_load_b128 v[145:148], off, off                     // 00000000918C: DC5D0000 917C0000 
	scratch_load_b128 v[149:152], off, off offset:16           // 000000009194: DC5D0010 957C0000 
	s_waitcnt vmcnt(1)                                         // 00000000919C: BF8907F7 
	ds_store_b16_d16_hi v32, v145                              // 0000000091A0: DA840000 00009120 
	ds_store_b16_d16_hi v32, v146 offset:64                    // 0000000091A8: DA840040 00009220 
	ds_store_b16_d16_hi v32, v147 offset:128                   // 0000000091B0: DA840080 00009320 
	ds_store_b16_d16_hi v32, v148 offset:192                   // 0000000091B8: DA8400C0 00009420 
	s_waitcnt vmcnt(0)                                         // 0000000091C0: BF8903F7 
	ds_store_b16_d16_hi v32, v149 offset:256                   // 0000000091C4: DA840100 00009520 
	ds_store_b16_d16_hi v32, v150 offset:320                   // 0000000091CC: DA840140 00009620 
	ds_store_b16_d16_hi v32, v151 offset:384                   // 0000000091D4: DA840180 00009720 
	ds_store_b16_d16_hi v32, v152 offset:448                   // 0000000091DC: DA8401C0 00009820 
	s_waitcnt lgkmcnt(0)                                       // 0000000091E4: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000091E8: BE83206A 
	s_cbranch_execz 77                                         // 0000000091EC: BFA5004D  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v41 offset:2048                        // 0000000091F0: DA980800 02000029 
	v_add_nc_u32_e32 v0, 0x800, v41                            // 0000000091F8: 4A0052FF 00000800 
	v_add_nc_u32_e32 v1, 0xa00, v41                            // 000000009200: 4A0252FF 00000A00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009208: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000009214: BF89FC07 
	ds_store_b16 v41, v2 offset:2560                           // 000000009218: D87C0A00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009220: 8B7E007E 
	s_cbranch_execz 63                                         // 000000009224: BFA5003F  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v0 offset:64                           // 000000009228: DA980040 02000000 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000009230: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000923C: BF89FC07 
	ds_store_b16 v1, v2 offset:64                              // 000000009240: D87C0040 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009248: 8B7E007E 
	s_cbranch_execz 53                                         // 00000000924C: BFA50035  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v0 offset:128                          // 000000009250: DA980080 02000000 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000009258: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009264: BF89FC07 
	ds_store_b16 v1, v2 offset:128                             // 000000009268: D87C0080 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009270: 8B7E007E 
	s_cbranch_execz 43                                         // 000000009274: BFA5002B  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009278: DA9800C0 02000000 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000009280: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 00000000928C: BF89FC07 
	ds_store_b16 v1, v2 offset:192                             // 000000009290: D87C00C0 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009298: 8B7E007E 
	s_cbranch_execz 33                                         // 00000000929C: BFA50021  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v0 offset:256                          // 0000000092A0: DA980100 02000000 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000092A8: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000092B4: BF89FC07 
	ds_store_b16 v1, v2 offset:256                             // 0000000092B8: D87C0100 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000092C0: 8B7E007E 
	s_cbranch_execz 23                                         // 0000000092C4: BFA50017  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v0 offset:320                          // 0000000092C8: DA980140 02000000 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000092D0: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000092D8: BF89FC07 
	ds_store_b16 v1, v2 offset:320                             // 0000000092DC: D87C0140 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000092E4: 8B7E007E 
	s_cbranch_execz 14                                         // 0000000092E8: BFA5000E  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v2, v0 offset:384                          // 0000000092EC: DA980180 02000000 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000092F4: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000092FC: BF89FC07 
	ds_store_b16 v1, v2 offset:384                             // 000000009300: D87C0180 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009308: 8B7E007E 
	s_cbranch_execz 5                                          // 00000000930C: BFA50005  <_amdgpu_cs_main+0x9324>
	ds_load_u16_d16 v0, v0 offset:448                          // 000000009310: DA9801C0 00000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009318: BF89FC07 
	ds_store_b16 v1, v0 offset:448                             // 00000000931C: D87C01C0 00000001 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009324: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000009328: BF89FC07 
	ds_store_b16_d16_hi v32, v208                              // 00000000932C: DA840000 0000D020 
	ds_store_b16_d16_hi v32, v209 offset:64                    // 000000009334: DA840040 0000D120 
	ds_store_b16_d16_hi v32, v210 offset:128                   // 00000000933C: DA840080 0000D220 
	ds_store_b16_d16_hi v32, v211 offset:192                   // 000000009344: DA8400C0 0000D320 
	ds_store_b16_d16_hi v32, v212 offset:256                   // 00000000934C: DA840100 0000D420 
	ds_store_b16_d16_hi v32, v213 offset:320                   // 000000009354: DA840140 0000D520 
	ds_store_b16_d16_hi v32, v214 offset:384                   // 00000000935C: DA840180 0000D620 
	ds_store_b16_d16_hi v32, v215 offset:448                   // 000000009364: DA8401C0 0000D720 
	s_waitcnt lgkmcnt(0)                                       // 00000000936C: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009370: BE83206A 
	s_cbranch_execz 108                                        // 000000009374: BFA5006C  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v41 offset:2560                        // 000000009378: DA980A00 02000029 
	s_waitcnt lgkmcnt(0)                                       // 000000009380: BF89FC07 
	ds_load_u16_d16_hi v2, v41 offset:2048                     // 000000009384: DA9C0800 02000029 
	v_add_nc_u32_e32 v0, 0xa00, v41                            // 00000000938C: 4A0052FF 00000A00 
	v_add_nc_u32_e32 v1, 0x800, v41                            // 000000009394: 4A0252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 00000000939C: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000093A8: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 0000000093AC: 64040582 
	ds_store_b16 v41, v2 offset:2560                           // 0000000093B0: D87C0A00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000093B8: 8B7E007E 
	s_cbranch_execz 90                                         // 0000000093BC: BFA5005A  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:64                           // 0000000093C0: DA980040 02000000 
	s_waitcnt lgkmcnt(0)                                       // 0000000093C8: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:64                        // 0000000093CC: DA9C0040 02000001 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000093D4: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000093E0: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 0000000093E4: 64040582 
	ds_store_b16 v0, v2 offset:64                              // 0000000093E8: D87C0040 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000093F0: 8B7E007E 
	s_cbranch_execz 76                                         // 0000000093F4: BFA5004C  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:128                          // 0000000093F8: DA980080 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009400: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:128                       // 000000009404: DA9C0080 02000001 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 00000000940C: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009418: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000941C: 64040582 
	ds_store_b16 v0, v2 offset:128                             // 000000009420: D87C0080 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009428: 8B7E007E 
	s_cbranch_execz 62                                         // 00000000942C: BFA5003E  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009430: DA9800C0 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009438: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:192                       // 00000000943C: DA9C00C0 02000001 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000009444: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000009450: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009454: 64040582 
	ds_store_b16 v0, v2 offset:192                             // 000000009458: D87C00C0 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009460: 8B7E007E 
	s_cbranch_execz 48                                         // 000000009464: BFA50030  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:256                          // 000000009468: DA980100 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009470: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:256                       // 000000009474: DA9C0100 02000001 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 00000000947C: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000009488: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000948C: 64040582 
	ds_store_b16 v0, v2 offset:256                             // 000000009490: D87C0100 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009498: 8B7E007E 
	s_cbranch_execz 34                                         // 00000000949C: BFA50022  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:320                          // 0000000094A0: DA980140 02000000 
	s_waitcnt lgkmcnt(0)                                       // 0000000094A8: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:320                       // 0000000094AC: DA9C0140 02000001 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000094B4: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000094BC: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 0000000094C0: 64040582 
	ds_store_b16 v0, v2 offset:320                             // 0000000094C4: D87C0140 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000094CC: 8B7E007E 
	s_cbranch_execz 21                                         // 0000000094D0: BFA50015  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:384                          // 0000000094D4: DA980180 02000000 
	s_waitcnt lgkmcnt(0)                                       // 0000000094DC: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:384                       // 0000000094E0: DA9C0180 02000001 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 0000000094E8: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000094F0: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 0000000094F4: 64040582 
	ds_store_b16 v0, v2 offset:384                             // 0000000094F8: D87C0180 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009500: 8B7E007E 
	s_cbranch_execz 8                                          // 000000009504: BFA50008  <_amdgpu_cs_main+0x9528>
	ds_load_u16_d16 v2, v0 offset:448                          // 000000009508: DA9801C0 02000000 
	ds_load_u16_d16 v1, v1 offset:448                          // 000000009510: DA9801C0 01000001 
	s_waitcnt lgkmcnt(0)                                       // 000000009518: BF89FC07 
	v_add_f16_e32 v1, v1, v2                                   // 00000000951C: 64020501 
	ds_store_b16 v0, v1 offset:448                             // 000000009520: D87C01C0 00000100 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009528: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 00000000952C: BF89FC07 
	ds_store_b16_d16_hi v32, v50                               // 000000009530: DA840000 00003220 
	ds_store_b16_d16_hi v32, v51 offset:64                     // 000000009538: DA840040 00003320 
	ds_store_b16_d16_hi v32, v52 offset:128                    // 000000009540: DA840080 00003420 
	ds_store_b16_d16_hi v32, v53 offset:192                    // 000000009548: DA8400C0 00003520 
	ds_store_b16_d16_hi v32, v54 offset:256                    // 000000009550: DA840100 00003620 
	ds_store_b16_d16_hi v32, v55 offset:320                    // 000000009558: DA840140 00003720 
	ds_store_b16_d16_hi v32, v56 offset:384                    // 000000009560: DA840180 00003820 
	ds_store_b16_d16_hi v32, v57 offset:448                    // 000000009568: DA8401C0 00003920 
	s_waitcnt lgkmcnt(0)                                       // 000000009570: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009574: BE83206A 
	s_cbranch_execz 77                                         // 000000009578: BFA5004D  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v41 offset:2048                        // 00000000957C: DA980800 02000029 
	v_add_nc_u32_e32 v0, 0x800, v41                            // 000000009584: 4A0052FF 00000800 
	v_add_nc_u32_e32 v1, 0xc00, v41                            // 00000000958C: 4A0252FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009594: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 0000000095A0: BF89FC07 
	ds_store_b16 v41, v2 offset:3072                           // 0000000095A4: D87C0C00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000095AC: 8B7E007E 
	s_cbranch_execz 63                                         // 0000000095B0: BFA5003F  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v0 offset:64                           // 0000000095B4: DA980040 02000000 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 0000000095BC: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000095C8: BF89FC07 
	ds_store_b16 v1, v2 offset:64                              // 0000000095CC: D87C0040 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000095D4: 8B7E007E 
	s_cbranch_execz 53                                         // 0000000095D8: BFA50035  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v0 offset:128                          // 0000000095DC: DA980080 02000000 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 0000000095E4: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000095F0: BF89FC07 
	ds_store_b16 v1, v2 offset:128                             // 0000000095F4: D87C0080 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000095FC: 8B7E007E 
	s_cbranch_execz 43                                         // 000000009600: BFA5002B  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009604: DA9800C0 02000000 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000960C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000009618: BF89FC07 
	ds_store_b16 v1, v2 offset:192                             // 00000000961C: D87C00C0 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009624: 8B7E007E 
	s_cbranch_execz 33                                         // 000000009628: BFA50021  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v0 offset:256                          // 00000000962C: DA980100 02000000 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000009634: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000009640: BF89FC07 
	ds_store_b16 v1, v2 offset:256                             // 000000009644: D87C0100 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000964C: 8B7E007E 
	s_cbranch_execz 23                                         // 000000009650: BFA50017  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v0 offset:320                          // 000000009654: DA980140 02000000 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000965C: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009664: BF89FC07 
	ds_store_b16 v1, v2 offset:320                             // 000000009668: D87C0140 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009670: 8B7E007E 
	s_cbranch_execz 14                                         // 000000009674: BFA5000E  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v2, v0 offset:384                          // 000000009678: DA980180 02000000 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000009680: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009688: BF89FC07 
	ds_store_b16 v1, v2 offset:384                             // 00000000968C: D87C0180 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009694: 8B7E007E 
	s_cbranch_execz 5                                          // 000000009698: BFA50005  <_amdgpu_cs_main+0x96B0>
	ds_load_u16_d16 v0, v0 offset:448                          // 00000000969C: DA9801C0 00000000 
	s_waitcnt lgkmcnt(0)                                       // 0000000096A4: BF89FC07 
	ds_store_b16 v1, v0 offset:448                             // 0000000096A8: D87C01C0 00000001 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000096B0: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000096B4: BF89FC07 
	ds_store_b16_d16_hi v32, v58                               // 0000000096B8: DA840000 00003A20 
	ds_store_b16_d16_hi v32, v59 offset:64                     // 0000000096C0: DA840040 00003B20 
	ds_store_b16_d16_hi v32, v60 offset:128                    // 0000000096C8: DA840080 00003C20 
	ds_store_b16_d16_hi v32, v61 offset:192                    // 0000000096D0: DA8400C0 00003D20 
	ds_store_b16_d16_hi v32, v62 offset:256                    // 0000000096D8: DA840100 00003E20 
	ds_store_b16_d16_hi v32, v63 offset:320                    // 0000000096E0: DA840140 00003F20 
	ds_store_b16_d16_hi v32, v64 offset:384                    // 0000000096E8: DA840180 00004020 
	ds_store_b16_d16_hi v32, v65 offset:448                    // 0000000096F0: DA8401C0 00004120 
	s_waitcnt lgkmcnt(0)                                       // 0000000096F8: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000096FC: BE83206A 
	s_cbranch_execz 108                                        // 000000009700: BFA5006C  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v41 offset:3072                        // 000000009704: DA980C00 02000029 
	s_waitcnt lgkmcnt(0)                                       // 00000000970C: BF89FC07 
	ds_load_u16_d16_hi v2, v41 offset:2048                     // 000000009710: DA9C0800 02000029 
	v_add_nc_u32_e32 v0, 0xc00, v41                            // 000000009718: 4A0052FF 00000C00 
	v_add_nc_u32_e32 v1, 0x800, v41                            // 000000009720: 4A0252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009728: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000009734: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009738: 64040582 
	ds_store_b16 v41, v2 offset:3072                           // 00000000973C: D87C0C00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009744: 8B7E007E 
	s_cbranch_execz 90                                         // 000000009748: BFA5005A  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:64                           // 00000000974C: DA980040 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009754: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:64                        // 000000009758: DA9C0040 02000001 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000009760: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000976C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009770: 64040582 
	ds_store_b16 v0, v2 offset:64                              // 000000009774: D87C0040 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000977C: 8B7E007E 
	s_cbranch_execz 76                                         // 000000009780: BFA5004C  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:128                          // 000000009784: DA980080 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000978C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:128                       // 000000009790: DA9C0080 02000001 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000009798: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 0000000097A4: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 0000000097A8: 64040582 
	ds_store_b16 v0, v2 offset:128                             // 0000000097AC: D87C0080 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000097B4: 8B7E007E 
	s_cbranch_execz 62                                         // 0000000097B8: BFA5003E  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:192                          // 0000000097BC: DA9800C0 02000000 
	s_waitcnt lgkmcnt(0)                                       // 0000000097C4: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:192                       // 0000000097C8: DA9C00C0 02000001 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 0000000097D0: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 0000000097DC: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 0000000097E0: 64040582 
	ds_store_b16 v0, v2 offset:192                             // 0000000097E4: D87C00C0 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000097EC: 8B7E007E 
	s_cbranch_execz 48                                         // 0000000097F0: BFA50030  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:256                          // 0000000097F4: DA980100 02000000 
	s_waitcnt lgkmcnt(0)                                       // 0000000097FC: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:256                       // 000000009800: DA9C0100 02000001 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000009808: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000009814: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009818: 64040582 
	ds_store_b16 v0, v2 offset:256                             // 00000000981C: D87C0100 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009824: 8B7E007E 
	s_cbranch_execz 34                                         // 000000009828: BFA50022  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:320                          // 00000000982C: DA980140 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009834: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:320                       // 000000009838: DA9C0140 02000001 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000009840: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009848: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000984C: 64040582 
	ds_store_b16 v0, v2 offset:320                             // 000000009850: D87C0140 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009858: 8B7E007E 
	s_cbranch_execz 21                                         // 00000000985C: BFA50015  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:384                          // 000000009860: DA980180 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009868: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:384                       // 00000000986C: DA9C0180 02000001 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000009874: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000987C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009880: 64040582 
	ds_store_b16 v0, v2 offset:384                             // 000000009884: D87C0180 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000988C: 8B7E007E 
	s_cbranch_execz 8                                          // 000000009890: BFA50008  <_amdgpu_cs_main+0x98B4>
	ds_load_u16_d16 v2, v0 offset:448                          // 000000009894: DA9801C0 02000000 
	ds_load_u16_d16 v1, v1 offset:448                          // 00000000989C: DA9801C0 01000001 
	s_waitcnt lgkmcnt(0)                                       // 0000000098A4: BF89FC07 
	v_add_f16_e32 v1, v1, v2                                   // 0000000098A8: 64020501 
	ds_store_b16 v0, v1 offset:448                             // 0000000098AC: D87C01C0 00000100 
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000098B4: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 0000000098B8: BF89FC07 
	ds_store_b16_d16_hi v32, v105                              // 0000000098BC: DA840000 00006920 
	ds_store_b16_d16_hi v32, v106 offset:64                    // 0000000098C4: DA840040 00006A20 
	ds_store_b16_d16_hi v32, v107 offset:128                   // 0000000098CC: DA840080 00006B20 
	ds_store_b16_d16_hi v32, v108 offset:192                   // 0000000098D4: DA8400C0 00006C20 
	ds_store_b16_d16_hi v32, v109 offset:256                   // 0000000098DC: DA840100 00006D20 
	ds_store_b16_d16_hi v32, v110 offset:320                   // 0000000098E4: DA840140 00006E20 
	ds_store_b16_d16_hi v32, v111 offset:384                   // 0000000098EC: DA840180 00006F20 
	ds_store_b16_d16_hi v32, v112 offset:448                   // 0000000098F4: DA8401C0 00007020 
	s_waitcnt lgkmcnt(0)                                       // 0000000098FC: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009900: BE83206A 
	s_cbranch_execz 77                                         // 000000009904: BFA5004D  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v41 offset:2048                        // 000000009908: DA980800 02000029 
	v_add_nc_u32_e32 v0, 0x800, v41                            // 000000009910: 4A0052FF 00000800 
	v_add_nc_u32_e32 v1, 0xe00, v41                            // 000000009918: 4A0252FF 00000E00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009920: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 00000000992C: BF89FC07 
	ds_store_b16 v41, v2 offset:3584                           // 000000009930: D87C0E00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009938: 8B7E007E 
	s_cbranch_execz 63                                         // 00000000993C: BFA5003F  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v0 offset:64                           // 000000009940: DA980040 02000000 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000009948: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009954: BF89FC07 
	ds_store_b16 v1, v2 offset:64                              // 000000009958: D87C0040 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009960: 8B7E007E 
	s_cbranch_execz 53                                         // 000000009964: BFA50035  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v0 offset:128                          // 000000009968: DA980080 02000000 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000009970: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000997C: BF89FC07 
	ds_store_b16 v1, v2 offset:128                             // 000000009980: D87C0080 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009988: 8B7E007E 
	s_cbranch_execz 43                                         // 00000000998C: BFA5002B  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009990: DA9800C0 02000000 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000009998: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 0000000099A4: BF89FC07 
	ds_store_b16 v1, v2 offset:192                             // 0000000099A8: D87C00C0 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000099B0: 8B7E007E 
	s_cbranch_execz 33                                         // 0000000099B4: BFA50021  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v0 offset:256                          // 0000000099B8: DA980100 02000000 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 0000000099C0: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 0000000099CC: BF89FC07 
	ds_store_b16 v1, v2 offset:256                             // 0000000099D0: D87C0100 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000099D8: 8B7E007E 
	s_cbranch_execz 23                                         // 0000000099DC: BFA50017  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v0 offset:320                          // 0000000099E0: DA980140 02000000 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 0000000099E8: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 0000000099F0: BF89FC07 
	ds_store_b16 v1, v2 offset:320                             // 0000000099F4: D87C0140 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 0000000099FC: 8B7E007E 
	s_cbranch_execz 14                                         // 000000009A00: BFA5000E  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v2, v0 offset:384                          // 000000009A04: DA980180 02000000 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000009A0C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009A14: BF89FC07 
	ds_store_b16 v1, v2 offset:384                             // 000000009A18: D87C0180 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009A20: 8B7E007E 
	s_cbranch_execz 5                                          // 000000009A24: BFA50005  <_amdgpu_cs_main+0x9A3C>
	ds_load_u16_d16 v0, v0 offset:448                          // 000000009A28: DA9801C0 00000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009A30: BF89FC07 
	ds_store_b16 v1, v0 offset:448                             // 000000009A34: D87C01C0 00000001 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009A3C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000009A40: BF89FC07 
	ds_store_b16_d16_hi v32, v113                              // 000000009A44: DA840000 00007120 
	ds_store_b16_d16_hi v32, v114 offset:64                    // 000000009A4C: DA840040 00007220 
	ds_store_b16_d16_hi v32, v115 offset:128                   // 000000009A54: DA840080 00007320 
	ds_store_b16_d16_hi v32, v116 offset:192                   // 000000009A5C: DA8400C0 00007420 
	ds_store_b16_d16_hi v32, v117 offset:256                   // 000000009A64: DA840100 00007520 
	ds_store_b16_d16_hi v32, v118 offset:320                   // 000000009A6C: DA840140 00007620 
	ds_store_b16_d16_hi v32, v119 offset:384                   // 000000009A74: DA840180 00007720 
	ds_store_b16_d16_hi v32, v120 offset:448                   // 000000009A7C: DA8401C0 00007820 
	s_waitcnt lgkmcnt(0)                                       // 000000009A84: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009A88: BE83206A 
	s_cbranch_execz 108                                        // 000000009A8C: BFA5006C  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v41 offset:3584                        // 000000009A90: DA980E00 02000029 
	s_waitcnt lgkmcnt(0)                                       // 000000009A98: BF89FC07 
	ds_load_u16_d16_hi v2, v41 offset:2048                     // 000000009A9C: DA9C0800 02000029 
	v_add_nc_u32_e32 v0, 0xe00, v41                            // 000000009AA4: 4A0052FF 00000E00 
	v_add_nc_u32_e32 v1, 0x800, v41                            // 000000009AAC: 4A0252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009AB4: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000009AC0: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009AC4: 64040582 
	ds_store_b16 v41, v2 offset:3584                           // 000000009AC8: D87C0E00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009AD0: 8B7E007E 
	s_cbranch_execz 90                                         // 000000009AD4: BFA5005A  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:64                           // 000000009AD8: DA980040 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009AE0: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:64                        // 000000009AE4: DA9C0040 02000001 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000009AEC: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009AF8: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009AFC: 64040582 
	ds_store_b16 v0, v2 offset:64                              // 000000009B00: D87C0040 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009B08: 8B7E007E 
	s_cbranch_execz 76                                         // 000000009B0C: BFA5004C  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:128                          // 000000009B10: DA980080 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009B18: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:128                       // 000000009B1C: DA9C0080 02000001 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000009B24: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009B30: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009B34: 64040582 
	ds_store_b16 v0, v2 offset:128                             // 000000009B38: D87C0080 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009B40: 8B7E007E 
	s_cbranch_execz 62                                         // 000000009B44: BFA5003E  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009B48: DA9800C0 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009B50: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:192                       // 000000009B54: DA9C00C0 02000001 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000009B5C: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000009B68: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009B6C: 64040582 
	ds_store_b16 v0, v2 offset:192                             // 000000009B70: D87C00C0 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009B78: 8B7E007E 
	s_cbranch_execz 48                                         // 000000009B7C: BFA50030  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:256                          // 000000009B80: DA980100 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009B88: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:256                       // 000000009B8C: DA9C0100 02000001 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000009B94: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000009BA0: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009BA4: 64040582 
	ds_store_b16 v0, v2 offset:256                             // 000000009BA8: D87C0100 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009BB0: 8B7E007E 
	s_cbranch_execz 34                                         // 000000009BB4: BFA50022  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:320                          // 000000009BB8: DA980140 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009BC0: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:320                       // 000000009BC4: DA9C0140 02000001 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000009BCC: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009BD4: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009BD8: 64040582 
	ds_store_b16 v0, v2 offset:320                             // 000000009BDC: D87C0140 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009BE4: 8B7E007E 
	s_cbranch_execz 21                                         // 000000009BE8: BFA50015  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:384                          // 000000009BEC: DA980180 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009BF4: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:384                       // 000000009BF8: DA9C0180 02000001 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000009C00: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009C08: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009C0C: 64040582 
	ds_store_b16 v0, v2 offset:384                             // 000000009C10: D87C0180 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009C18: 8B7E007E 
	s_cbranch_execz 8                                          // 000000009C1C: BFA50008  <_amdgpu_cs_main+0x9C40>
	ds_load_u16_d16 v2, v0 offset:448                          // 000000009C20: DA9801C0 02000000 
	ds_load_u16_d16 v1, v1 offset:448                          // 000000009C28: DA9801C0 01000001 
	s_waitcnt lgkmcnt(0)                                       // 000000009C30: BF89FC07 
	v_add_f16_e32 v1, v1, v2                                   // 000000009C34: 64020501 
	ds_store_b16 v0, v1 offset:448                             // 000000009C38: D87C01C0 00000100 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009C40: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000009C44: BF89FC07 
	ds_store_b16_d16_hi v32, v168                              // 000000009C48: DA840000 0000A820 
	ds_store_b16_d16_hi v32, v169 offset:64                    // 000000009C50: DA840040 0000A920 
	ds_store_b16_d16_hi v32, v170 offset:128                   // 000000009C58: DA840080 0000AA20 
	ds_store_b16_d16_hi v32, v171 offset:192                   // 000000009C60: DA8400C0 0000AB20 
	ds_store_b16_d16_hi v32, v172 offset:256                   // 000000009C68: DA840100 0000AC20 
	ds_store_b16_d16_hi v32, v173 offset:320                   // 000000009C70: DA840140 0000AD20 
	ds_store_b16_d16_hi v32, v174 offset:384                   // 000000009C78: DA840180 0000AE20 
	ds_store_b16_d16_hi v32, v175 offset:448                   // 000000009C80: DA8401C0 0000AF20 
	s_waitcnt lgkmcnt(0)                                       // 000000009C88: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009C8C: BE83206A 
	s_cbranch_execz 77                                         // 000000009C90: BFA5004D  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v41 offset:2048                        // 000000009C94: DA980800 02000029 
	v_add_nc_u32_e32 v0, 0x800, v41                            // 000000009C9C: 4A0052FF 00000800 
	v_add_nc_u32_e32 v1, 0x1000, v41                           // 000000009CA4: 4A0252FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009CAC: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000009CB8: BF89FC07 
	ds_store_b16 v41, v2 offset:4096                           // 000000009CBC: D87C1000 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009CC4: 8B7E007E 
	s_cbranch_execz 63                                         // 000000009CC8: BFA5003F  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v0 offset:64                           // 000000009CCC: DA980040 02000000 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000009CD4: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009CE0: BF89FC07 
	ds_store_b16 v1, v2 offset:64                              // 000000009CE4: D87C0040 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009CEC: 8B7E007E 
	s_cbranch_execz 53                                         // 000000009CF0: BFA50035  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v0 offset:128                          // 000000009CF4: DA980080 02000000 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000009CFC: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009D08: BF89FC07 
	ds_store_b16 v1, v2 offset:128                             // 000000009D0C: D87C0080 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009D14: 8B7E007E 
	s_cbranch_execz 43                                         // 000000009D18: BFA5002B  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009D1C: DA9800C0 02000000 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000009D24: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000009D30: BF89FC07 
	ds_store_b16 v1, v2 offset:192                             // 000000009D34: D87C00C0 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009D3C: 8B7E007E 
	s_cbranch_execz 33                                         // 000000009D40: BFA50021  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v0 offset:256                          // 000000009D44: DA980100 02000000 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000009D4C: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000009D58: BF89FC07 
	ds_store_b16 v1, v2 offset:256                             // 000000009D5C: D87C0100 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009D64: 8B7E007E 
	s_cbranch_execz 23                                         // 000000009D68: BFA50017  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v0 offset:320                          // 000000009D6C: DA980140 02000000 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000009D74: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009D7C: BF89FC07 
	ds_store_b16 v1, v2 offset:320                             // 000000009D80: D87C0140 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009D88: 8B7E007E 
	s_cbranch_execz 14                                         // 000000009D8C: BFA5000E  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v2, v0 offset:384                          // 000000009D90: DA980180 02000000 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000009D98: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009DA0: BF89FC07 
	ds_store_b16 v1, v2 offset:384                             // 000000009DA4: D87C0180 00000201 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009DAC: 8B7E007E 
	s_cbranch_execz 5                                          // 000000009DB0: BFA50005  <_amdgpu_cs_main+0x9DC8>
	ds_load_u16_d16 v0, v0 offset:448                          // 000000009DB4: DA9801C0 00000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009DBC: BF89FC07 
	ds_store_b16 v1, v0 offset:448                             // 000000009DC0: D87C01C0 00000001 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009DC8: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000009DCC: BF89FC07 
	ds_store_b16_d16_hi v32, v160                              // 000000009DD0: DA840000 0000A020 
	ds_store_b16_d16_hi v32, v161 offset:64                    // 000000009DD8: DA840040 0000A120 
	ds_store_b16_d16_hi v32, v162 offset:128                   // 000000009DE0: DA840080 0000A220 
	ds_store_b16_d16_hi v32, v163 offset:192                   // 000000009DE8: DA8400C0 0000A320 
	ds_store_b16_d16_hi v32, v164 offset:256                   // 000000009DF0: DA840100 0000A420 
	ds_store_b16_d16_hi v32, v165 offset:320                   // 000000009DF8: DA840140 0000A520 
	ds_store_b16_d16_hi v32, v166 offset:384                   // 000000009E00: DA840180 0000A620 
	ds_store_b16_d16_hi v32, v167 offset:448                   // 000000009E08: DA8401C0 0000A720 
	s_waitcnt lgkmcnt(0)                                       // 000000009E10: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009E14: BE83206A 
	s_cbranch_execz 108                                        // 000000009E18: BFA5006C  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v41 offset:4096                        // 000000009E1C: DA981000 02000029 
	s_waitcnt lgkmcnt(0)                                       // 000000009E24: BF89FC07 
	ds_load_u16_d16_hi v2, v41 offset:2048                     // 000000009E28: DA9C0800 02000029 
	v_add_nc_u32_e32 v0, 0x1000, v41                           // 000000009E30: 4A0052FF 00001000 
	v_add_nc_u32_e32 v1, 0x800, v41                            // 000000009E38: 4A0252FF 00000800 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 000000009E40: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 000000009E4C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009E50: 64040582 
	ds_store_b16 v41, v2 offset:4096                           // 000000009E54: D87C1000 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009E5C: 8B7E007E 
	s_cbranch_execz 90                                         // 000000009E60: BFA5005A  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:64                           // 000000009E64: DA980040 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009E6C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:64                        // 000000009E70: DA9C0040 02000001 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 000000009E78: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009E84: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009E88: 64040582 
	ds_store_b16 v0, v2 offset:64                              // 000000009E8C: D87C0040 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009E94: 8B7E007E 
	s_cbranch_execz 76                                         // 000000009E98: BFA5004C  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:128                          // 000000009E9C: DA980080 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009EA4: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:128                       // 000000009EA8: DA9C0080 02000001 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 000000009EB0: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009EBC: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009EC0: 64040582 
	ds_store_b16 v0, v2 offset:128                             // 000000009EC4: D87C0080 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009ECC: 8B7E007E 
	s_cbranch_execz 62                                         // 000000009ED0: BFA5003E  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:192                          // 000000009ED4: DA9800C0 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009EDC: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:192                       // 000000009EE0: DA9C00C0 02000001 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 000000009EE8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 000000009EF4: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009EF8: 64040582 
	ds_store_b16 v0, v2 offset:192                             // 000000009EFC: D87C00C0 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009F04: 8B7E007E 
	s_cbranch_execz 48                                         // 000000009F08: BFA50030  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:256                          // 000000009F0C: DA980100 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009F14: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:256                       // 000000009F18: DA9C0100 02000001 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 000000009F20: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 000000009F2C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009F30: 64040582 
	ds_store_b16 v0, v2 offset:256                             // 000000009F34: D87C0100 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009F3C: 8B7E007E 
	s_cbranch_execz 34                                         // 000000009F40: BFA50022  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:320                          // 000000009F44: DA980140 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009F4C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:320                       // 000000009F50: DA9C0140 02000001 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 000000009F58: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 000000009F60: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009F64: 64040582 
	ds_store_b16 v0, v2 offset:320                             // 000000009F68: D87C0140 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009F70: 8B7E007E 
	s_cbranch_execz 21                                         // 000000009F74: BFA50015  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:384                          // 000000009F78: DA980180 02000000 
	s_waitcnt lgkmcnt(0)                                       // 000000009F80: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:384                       // 000000009F84: DA9C0180 02000001 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 000000009F8C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 000000009F94: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 000000009F98: 64040582 
	ds_store_b16 v0, v2 offset:384                             // 000000009F9C: D87C0180 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 000000009FA4: 8B7E007E 
	s_cbranch_execz 8                                          // 000000009FA8: BFA50008  <_amdgpu_cs_main+0x9FCC>
	ds_load_u16_d16 v2, v0 offset:448                          // 000000009FAC: DA9801C0 02000000 
	ds_load_u16_d16 v1, v1 offset:448                          // 000000009FB4: DA9801C0 01000001 
	s_waitcnt lgkmcnt(0)                                       // 000000009FBC: BF89FC07 
	v_add_f16_e32 v1, v1, v2                                   // 000000009FC0: 64020501 
	ds_store_b16 v0, v1 offset:448                             // 000000009FC4: D87C01C0 00000100 
	s_or_b32 exec_lo, exec_lo, s3                              // 000000009FCC: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 000000009FD0: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 000000009FD4: BE83206A 
	s_cbranch_execz 108                                        // 000000009FD8: BFA5006C  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v41 offset:2560                        // 000000009FDC: DA980A00 02000029 
	s_waitcnt lgkmcnt(0)                                       // 000000009FE4: BF89FC07 
	ds_load_u16_d16_hi v2, v41 offset:3072                     // 000000009FE8: DA9C0C00 02000029 
	v_add_nc_u32_e32 v0, 0xa00, v41                            // 000000009FF0: 4A0052FF 00000A00 
	v_add_nc_u32_e32 v1, 0xc00, v41                            // 000000009FF8: 4A0252FF 00000C00 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 00000000A000: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A00C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A010: 64040582 
	ds_store_b16 v41, v2 offset:2560                           // 00000000A014: D87C0A00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A01C: 8B7E007E 
	s_cbranch_execz 90                                         // 00000000A020: BFA5005A  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:64                           // 00000000A024: DA980040 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A02C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:64                        // 00000000A030: DA9C0040 02000001 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 00000000A038: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A044: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A048: 64040582 
	ds_store_b16 v0, v2 offset:64                              // 00000000A04C: D87C0040 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A054: 8B7E007E 
	s_cbranch_execz 76                                         // 00000000A058: BFA5004C  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:128                          // 00000000A05C: DA980080 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A064: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:128                       // 00000000A068: DA9C0080 02000001 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 00000000A070: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A07C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A080: 64040582 
	ds_store_b16 v0, v2 offset:128                             // 00000000A084: D87C0080 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A08C: 8B7E007E 
	s_cbranch_execz 62                                         // 00000000A090: BFA5003E  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:192                          // 00000000A094: DA9800C0 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A09C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:192                       // 00000000A0A0: DA9C00C0 02000001 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000A0A8: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 00000000A0B4: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A0B8: 64040582 
	ds_store_b16 v0, v2 offset:192                             // 00000000A0BC: D87C00C0 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A0C4: 8B7E007E 
	s_cbranch_execz 48                                         // 00000000A0C8: BFA50030  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:256                          // 00000000A0CC: DA980100 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A0D4: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:256                       // 00000000A0D8: DA9C0100 02000001 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 00000000A0E0: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 00000000A0EC: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A0F0: 64040582 
	ds_store_b16 v0, v2 offset:256                             // 00000000A0F4: D87C0100 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A0FC: 8B7E007E 
	s_cbranch_execz 34                                         // 00000000A100: BFA50022  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:320                          // 00000000A104: DA980140 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A10C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:320                       // 00000000A110: DA9C0140 02000001 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000A118: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A120: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A124: 64040582 
	ds_store_b16 v0, v2 offset:320                             // 00000000A128: D87C0140 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A130: 8B7E007E 
	s_cbranch_execz 21                                         // 00000000A134: BFA50015  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:384                          // 00000000A138: DA980180 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A140: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:384                       // 00000000A144: DA9C0180 02000001 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 00000000A14C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A154: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A158: 64040582 
	ds_store_b16 v0, v2 offset:384                             // 00000000A15C: D87C0180 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A164: 8B7E007E 
	s_cbranch_execz 8                                          // 00000000A168: BFA50008  <_amdgpu_cs_main+0xA18C>
	ds_load_u16_d16 v2, v0 offset:448                          // 00000000A16C: DA9801C0 02000000 
	ds_load_u16_d16 v1, v1 offset:448                          // 00000000A174: DA9801C0 01000001 
	s_waitcnt lgkmcnt(0)                                       // 00000000A17C: BF89FC07 
	v_add_f16_e32 v1, v1, v2                                   // 00000000A180: 64020501 
	ds_store_b16 v0, v1 offset:448                             // 00000000A184: D87C01C0 00000100 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000A18C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 00000000A190: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 00000000A194: BE83206A 
	s_cbranch_execz 108                                        // 00000000A198: BFA5006C  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v41 offset:3584                        // 00000000A19C: DA980E00 02000029 
	s_waitcnt lgkmcnt(0)                                       // 00000000A1A4: BF89FC07 
	ds_load_u16_d16_hi v2, v41 offset:4096                     // 00000000A1A8: DA9C1000 02000029 
	v_add_nc_u32_e32 v0, 0xe00, v41                            // 00000000A1B0: 4A0052FF 00000E00 
	v_add_nc_u32_e32 v1, 0x1000, v41                           // 00000000A1B8: 4A0252FF 00001000 
	v_cmp_gt_u32_e64 s0, 0xe0, v129                            // 00000000A1C0: D44C0000 000302FF 000000E0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A1CC: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A1D0: 64040582 
	ds_store_b16 v41, v2 offset:3584                           // 00000000A1D4: D87C0E00 00000229 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A1DC: 8B7E007E 
	s_cbranch_execz 90                                         // 00000000A1E0: BFA5005A  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:64                           // 00000000A1E4: DA980040 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A1EC: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:64                        // 00000000A1F0: DA9C0040 02000001 
	v_cmp_gt_u32_e64 s0, 0xc0, v129                            // 00000000A1F8: D44C0000 000302FF 000000C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A204: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A208: 64040582 
	ds_store_b16 v0, v2 offset:64                              // 00000000A20C: D87C0040 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A214: 8B7E007E 
	s_cbranch_execz 76                                         // 00000000A218: BFA5004C  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:128                          // 00000000A21C: DA980080 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A224: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:128                       // 00000000A228: DA9C0080 02000001 
	v_cmp_gt_u32_e64 s0, 0xa0, v129                            // 00000000A230: D44C0000 000302FF 000000A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A23C: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A240: 64040582 
	ds_store_b16 v0, v2 offset:128                             // 00000000A244: D87C0080 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A24C: 8B7E007E 
	s_cbranch_execz 62                                         // 00000000A250: BFA5003E  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:192                          // 00000000A254: DA9800C0 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A25C: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:192                       // 00000000A260: DA9C00C0 02000001 
	v_cmp_gt_u32_e64 s0, 0x80, v129                            // 00000000A268: D44C0000 000302FF 00000080 
	s_waitcnt lgkmcnt(0)                                       // 00000000A274: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A278: 64040582 
	ds_store_b16 v0, v2 offset:192                             // 00000000A27C: D87C00C0 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A284: 8B7E007E 
	s_cbranch_execz 48                                         // 00000000A288: BFA50030  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:256                          // 00000000A28C: DA980100 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A294: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:256                       // 00000000A298: DA9C0100 02000001 
	v_cmp_gt_u32_e64 s0, 0x60, v129                            // 00000000A2A0: D44C0000 000302FF 00000060 
	s_waitcnt lgkmcnt(0)                                       // 00000000A2AC: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A2B0: 64040582 
	ds_store_b16 v0, v2 offset:256                             // 00000000A2B4: D87C0100 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A2BC: 8B7E007E 
	s_cbranch_execz 34                                         // 00000000A2C0: BFA50022  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:320                          // 00000000A2C4: DA980140 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A2CC: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:320                       // 00000000A2D0: DA9C0140 02000001 
	v_cmp_gt_u32_e64 s0, 64, v129                              // 00000000A2D8: D44C0000 000302C0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A2E0: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A2E4: 64040582 
	ds_store_b16 v0, v2 offset:320                             // 00000000A2E8: D87C0140 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A2F0: 8B7E007E 
	s_cbranch_execz 21                                         // 00000000A2F4: BFA50015  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:384                          // 00000000A2F8: DA980180 02000000 
	s_waitcnt lgkmcnt(0)                                       // 00000000A300: BF89FC07 
	ds_load_u16_d16_hi v2, v1 offset:384                       // 00000000A304: DA9C0180 02000001 
	v_cmp_gt_u32_e64 s0, 32, v129                              // 00000000A30C: D44C0000 000302A0 
	s_waitcnt lgkmcnt(0)                                       // 00000000A314: BF89FC07 
	v_add_f16_e32 v2, v130/*Invalid register, operand has 'VS_32_Lo128' register class*/, v2// 00000000A318: 64040582 
	ds_store_b16 v0, v2 offset:384                             // 00000000A31C: D87C0180 00000200 
	s_and_b32 exec_lo, exec_lo, s0                             // 00000000A324: 8B7E007E 
	s_cbranch_execz 8                                          // 00000000A328: BFA50008  <_amdgpu_cs_main+0xA34C>
	ds_load_u16_d16 v2, v0 offset:448                          // 00000000A32C: DA9801C0 02000000 
	ds_load_u16_d16 v1, v1 offset:448                          // 00000000A334: DA9801C0 01000001 
	s_waitcnt lgkmcnt(0)                                       // 00000000A33C: BF89FC07 
	v_add_f16_e32 v1, v1, v2                                   // 00000000A340: 64020501 
	ds_store_b16 v0, v1 offset:448                             // 00000000A344: D87C01C0 00000100 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000A34C: 8C7E037E 
	s_waitcnt lgkmcnt(0)                                       // 00000000A350: BF89FC07 
	s_and_saveexec_b32 s3, vcc_lo                              // 00000000A354: BE83206A 
	s_cbranch_execz 56                                         // 00000000A358: BFA50038  <_amdgpu_cs_main+0xA43C>
	v_add3_u32 v0, v34, s11, 16                                // 00000000A35C: D6550000 02401722 
	v_or3_b32 v2, s10, v128, 16                                // 00000000A364: D6580002 0243000A 
	s_mov_b32 s4, 0                                            // 00000000A36C: BE840080 
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A370: BF870112 
	v_mul_lo_u32 v1, s6, v0                                    // 00000000A374: D72C0001 00020006 
	v_cmp_gt_u32_e32 vcc_lo, s2, v2                            // 00000000A37C: 7C980402 
	v_add_nc_u32_e32 v2, 0xe00, v41                            // 00000000A380: 4A0452FF 00000E00 
	s_lshl_b32 s2, s6, 3                                       // 00000000A388: 84028306 
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000A38C: BF870123 
	v_add3_u32 v3, v1, s10, v128                               // 00000000A390: D6550003 06001501 
	v_add_nc_u32_e32 v1, 0xa00, v41                            // 00000000A398: 4A0252FF 00000A00 
	v_lshl_add_u32 v3, v3, 2, 64                               // 00000000A3A0: D6460003 03010503 
	s_set_inst_prefetch_distance 0x1                           // 00000000A3A8: BF840001 
	s_branch 17                                                // 00000000A3AC: BFA00011  <_amdgpu_cs_main+0xA3F4>
	s_nop 0                                                    // 00000000A3B0: BF800000 
	s_nop 0                                                    // 00000000A3B4: BF800000 
	s_nop 0                                                    // 00000000A3B8: BF800000 
	s_nop 0                                                    // 00000000A3BC: BF800000 
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000A3C0: 8C7E007E 
	v_add_nc_u32_e32 v33, 32, v33                              // 00000000A3C4: 4A4242A0 
	v_add_nc_u32_e32 v1, 64, v1                                // 00000000A3C8: 4A0202C0 
	v_add_nc_u32_e32 v2, 64, v2                                // 00000000A3CC: 4A0404C0 
	v_add_nc_u32_e32 v0, 2, v0                                 // 00000000A3D0: 4A000082 
	v_add_nc_u32_e32 v3, s2, v3                                // 00000000A3D4: 4A060602 
	v_cmp_lt_u32_e64 s0, 0xdf, v33                             // 00000000A3D8: D4490000 000242FF 000000DF 
	s_or_b32 s4, s0, s4                                        // 00000000A3E4: 8C040400 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A3E8: BF870009 
	s_and_not1_b32 exec_lo, exec_lo, s4                        // 00000000A3EC: 917E047E 
	s_cbranch_execz 18                                         // 00000000A3F0: BFA50012  <_amdgpu_cs_main+0xA43C>
	v_cmp_gt_u32_e64 s0, s1, v0                                // 00000000A3F4: D44C0000 00020001 
	s_and_b32 s5, s0, vcc_lo                                   // 00000000A3FC: 8B056A00 
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A400: BF870009 
	s_and_saveexec_b32 s0, s5                                  // 00000000A404: BE802005 
	s_cbranch_execz 65517                                      // 00000000A408: BFA5FFED  <_amdgpu_cs_main+0xA3C0>
	ds_load_u16_d16 v4, v1                                     // 00000000A40C: DA980000 04000001 
	s_waitcnt lgkmcnt(0)                                       // 00000000A414: BF89FC07 
	ds_load_u16_d16_hi v4, v2                                  // 00000000A418: DA9C0000 04000002 
	s_waitcnt lgkmcnt(0)                                       // 00000000A420: BF89FC07 
	v_add_f16_e32 v4, v132/*Invalid register, operand has 'VS_32_Lo128' register class*/, v4// 00000000A424: 64080984 
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000A428: BF870001 
	v_cvt_f32_f16_e32 v4, v4                                   // 00000000A42C: 7E081704 
	buffer_store_b32 v4, v3, s[20:23], 0 offen                 // 00000000A430: E0680000 80450403 
	s_branch 65505                                             // 00000000A438: BFA0FFE1  <_amdgpu_cs_main+0xA3C0>
	s_set_inst_prefetch_distance 0x2                           // 00000000A43C: BF840002 
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000A440: 8C7E037E 
	s_waitcnt_vscnt null, 0x0                                  // 00000000A444: BC7C0000 
	s_nop 0                                                    // 00000000A448: BF800000 
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000A44C: BFB60003 
	s_endpgm                                                   // 00000000A450: BFB00000 
	s_code_end                                                 // 00000000A454: BF9F0000 
	s_code_end                                                 // 00000000A458: BF9F0000 
	s_code_end                                                 // 00000000A45C: BF9F0000 
	s_code_end                                                 // 00000000A460: BF9F0000 
	s_code_end                                                 // 00000000A464: BF9F0000 
	s_code_end                                                 // 00000000A468: BF9F0000 
	s_code_end                                                 // 00000000A46C: BF9F0000 
	s_code_end                                                 // 00000000A470: BF9F0000 
	s_code_end                                                 // 00000000A474: BF9F0000 
	s_code_end                                                 // 00000000A478: BF9F0000 
	s_code_end                                                 // 00000000A47C: BF9F0000 
	s_code_end                                                 // 00000000A480: BF9F0000 
	s_code_end                                                 // 00000000A484: BF9F0000 
	s_code_end                                                 // 00000000A488: BF9F0000 
	s_code_end                                                 // 00000000A48C: BF9F0000 
	s_code_end                                                 // 00000000A490: BF9F0000 
	s_code_end                                                 // 00000000A494: BF9F0000 
	s_code_end                                                 // 00000000A498: BF9F0000 
	s_code_end                                                 // 00000000A49C: BF9F0000 
	s_code_end                                                 // 00000000A4A0: BF9F0000 
	s_code_end                                                 // 00000000A4A4: BF9F0000 
	s_code_end                                                 // 00000000A4A8: BF9F0000 
	s_code_end                                                 // 00000000A4AC: BF9F0000 
	s_code_end                                                 // 00000000A4B0: BF9F0000 
	s_code_end                                                 // 00000000A4B4: BF9F0000 
	s_code_end                                                 // 00000000A4B8: BF9F0000 
	s_code_end                                                 // 00000000A4BC: BF9F0000 
	s_code_end                                                 // 00000000A4C0: BF9F0000 
	s_code_end                                                 // 00000000A4C4: BF9F0000 
	s_code_end                                                 // 00000000A4C8: BF9F0000 
	s_code_end                                                 // 00000000A4CC: BF9F0000 
	s_code_end                                                 // 00000000A4D0: BF9F0000 
	s_code_end                                                 // 00000000A4D4: BF9F0000 
	s_code_end                                                 // 00000000A4D8: BF9F0000 
	s_code_end                                                 // 00000000A4DC: BF9F0000 
	s_code_end                                                 // 00000000A4E0: BF9F0000 
	s_code_end                                                 // 00000000A4E4: BF9F0000 
	s_code_end                                                 // 00000000A4E8: BF9F0000 
	s_code_end                                                 // 00000000A4EC: BF9F0000 
	s_code_end                                                 // 00000000A4F0: BF9F0000 
	s_code_end                                                 // 00000000A4F4: BF9F0000 
	s_code_end                                                 // 00000000A4F8: BF9F0000 
	s_code_end                                                 // 00000000A4FC: BF9F0000 
	s_code_end                                                 // 00000000A500: BF9F0000 
	s_code_end                                                 // 00000000A504: BF9F0000 
	s_code_end                                                 // 00000000A508: BF9F0000 
	s_code_end                                                 // 00000000A50C: BF9F0000 
	s_code_end                                                 // 00000000A510: BF9F0000 
	s_code_end                                                 // 00000000A514: BF9F0000 
	s_code_end                                                 // 00000000A518: BF9F0000 
	s_code_end                                                 // 00000000A51C: BF9F0000 
	s_code_end                                                 // 00000000A520: BF9F0000 
	s_code_end                                                 // 00000000A524: BF9F0000 
	s_code_end                                                 // 00000000A528: BF9F0000 
	s_code_end                                                 // 00000000A52C: BF9F0000 
	s_code_end                                                 // 00000000A530: BF9F0000 
	s_code_end                                                 // 00000000A534: BF9F0000 
	s_code_end                                                 // 00000000A538: BF9F0000 
	s_code_end                                                 // 00000000A53C: BF9F0000 
	s_code_end                                                 // 00000000A540: BF9F0000 
	s_code_end                                                 // 00000000A544: BF9F0000 
	s_code_end                                                 // 00000000A548: BF9F0000 
	s_code_end                                                 // 00000000A54C: BF9F0000 
	s_code_end                                                 // 00000000A550: BF9F0000 
	s_code_end                                                 // 00000000A554: BF9F0000 
	s_code_end                                                 // 00000000A558: BF9F0000 
	s_code_end                                                 // 00000000A55C: BF9F0000 
	s_code_end                                                 // 00000000A560: BF9F0000 
	s_code_end                                                 // 00000000A564: BF9F0000 
	s_code_end                                                 // 00000000A568: BF9F0000 
	s_code_end                                                 // 00000000A56C: BF9F0000 
	s_code_end                                                 // 00000000A570: BF9F0000 
	s_code_end                                                 // 00000000A574: BF9F0000 
	s_code_end                                                 // 00000000A578: BF9F0000 
	s_code_end                                                 // 00000000A57C: BF9F0000 
	s_code_end                                                 // 00000000A580: BF9F0000 
	s_code_end                                                 // 00000000A584: BF9F0000 
	s_code_end                                                 // 00000000A588: BF9F0000 
	s_code_end                                                 // 00000000A58C: BF9F0000 
	s_code_end                                                 // 00000000A590: BF9F0000 
	s_code_end                                                 // 00000000A594: BF9F0000 
	s_code_end                                                 // 00000000A598: BF9F0000 
	s_code_end                                                 // 00000000A59C: BF9F0000 
	s_code_end                                                 // 00000000A5A0: BF9F0000 
	s_code_end                                                 // 00000000A5A4: BF9F0000 
	s_code_end                                                 // 00000000A5A8: BF9F0000 
	s_code_end                                                 // 00000000A5AC: BF9F0000 
	s_code_end                                                 // 00000000A5B0: BF9F0000 
	s_code_end                                                 // 00000000A5B4: BF9F0000 
	s_code_end                                                 // 00000000A5B8: BF9F0000 
	s_code_end                                                 // 00000000A5BC: BF9F0000 
	s_code_end                                                 // 00000000A5C0: BF9F0000 
	s_code_end                                                 // 00000000A5C4: BF9F0000 
	s_code_end                                                 // 00000000A5C8: BF9F0000 
	s_code_end                                                 // 00000000A5CC: BF9F0000 
	s_code_end                                                 // 00000000A5D0: BF9F0000 
	s_code_end                                                 // 00000000A5D4: BF9F0000 
	s_code_end                                                 // 00000000A5D8: BF9F0000 
	s_code_end                                                 // 00000000A5DC: BF9F0000 
	s_code_end                                                 // 00000000A5E0: BF9F0000 
	s_code_end                                                 // 00000000A5E4: BF9F0000 
	s_code_end                                                 // 00000000A5E8: BF9F0000 
	s_code_end                                                 // 00000000A5EC: BF9F0000 
	s_code_end                                                 // 00000000A5F0: BF9F0000 
	s_code_end                                                 // 00000000A5F4: BF9F0000 
	s_code_end                                                 // 00000000A5F8: BF9F0000 
	s_code_end                                                 // 00000000A5FC: BF9F0000 
