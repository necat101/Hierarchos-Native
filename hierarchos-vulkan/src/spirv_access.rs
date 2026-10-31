//! Conservative storage-buffer access reflection for the command hazard tracker.
//!
//! SPIR-V NonWritable / NonReadable are module guarantees, including when they
//! decorate every member of a storage block. Unrecognized forms keep MayWrite.
//! https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Decoration

use std::collections::{HashMap, HashSet};

use super::BindingAccess;

pub(super) fn binding_accesses(words: &[u32], binding_count: usize) -> Vec<BindingAccess> {
    let conservative = || vec![BindingAccess::MayWrite; binding_count];
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return conservative();
    }
    let mut bindings = HashMap::new();
    let mut sets = HashMap::new();
    let mut non_writable = HashSet::new();
    let mut non_readable = HashSet::new();
    let mut member_non_writable = HashSet::new();
    let mut member_non_readable = HashSet::new();
    let mut struct_members = HashMap::new();
    let mut pointers = HashMap::new();
    let mut variables = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let opcode = words[offset] & 0xffff;
        if count == 0 || count > words.len() - offset {
            return conservative();
        }
        let operands = &words[offset + 1..offset + count];
        match (opcode, operands) {
            // OpDecorate: target, decoration, optional literal.
            (71, [target, 24]) => {
                non_writable.insert(*target);
            }
            (71, [target, 25]) => {
                non_readable.insert(*target);
            }
            (71, [target, 33, binding]) => {
                bindings.insert(*target, *binding);
            }
            (71, [target, 34, set]) => {
                sets.insert(*target, *set);
            }
            // OpMemberDecorate: struct, member, decoration.
            (72, [target, member, 24]) => {
                member_non_writable.insert((*target, *member));
            }
            (72, [target, member, 25]) => {
                member_non_readable.insert((*target, *member));
            }
            // OpTypeStruct: result, member types.
            (30, [target, members @ ..]) => {
                struct_members.insert(*target, members.len());
            }
            // OpTypePointer: result, storage class, pointee type.
            (32, [target, storage, pointee]) => {
                pointers.insert(*target, (*storage, *pointee));
            }
            // OpVariable: result type, result, storage class, optional initializer.
            (59, [pointer, target, storage, ..]) if *storage == 2 || *storage == 12 => {
                variables.push((*pointer, *target));
            }
            _ => {}
        }
        offset += count;
    }

    let mut accesses = vec![None; binding_count];
    for (pointer, target) in variables {
        if sets.get(&target) != Some(&0) {
            continue;
        }
        let Some(&binding) = bindings.get(&target) else {
            continue;
        };
        let Some(slot) = accesses.get_mut(binding as usize) else {
            continue;
        };
        let block = pointers.get(&pointer).map(|(_, pointee)| *pointee);
        let all_members_decorated = |decorations: &HashSet<(u32, u32)>| {
            block.is_some_and(|block| {
                struct_members.get(&block).is_some_and(|&count| {
                    count > 0
                        && (0..count).all(|member| decorations.contains(&(block, member as u32)))
                })
            })
        };
        let readonly = non_writable.contains(&target)
            || block.is_some_and(|block| non_writable.contains(&block))
            || all_members_decorated(&member_non_writable);
        let writeonly = non_readable.contains(&target)
            || block.is_some_and(|block| non_readable.contains(&block))
            || all_members_decorated(&member_non_readable);
        let access = match (readonly, writeonly) {
            (true, false) => BindingAccess::ReadOnly,
            (false, true) => BindingAccess::WriteOnly,
            _ => BindingAccess::MayWrite,
        };
        // Multiple variables can alias one binding. Only retain a restriction
        // when every declaration at that binding agrees with it.
        *slot = Some(match *slot {
            None => access,
            Some(previous) if previous == access => access,
            Some(_) => BindingAccess::MayWrite,
        });
    }
    accesses
        .into_iter()
        .map(|access| access.unwrap_or(BindingAccess::MayWrite))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflects_embedded_read_write_and_inplace_buffers() {
        let linear = ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(
            "../shaders/linear_forward.spv"
        )))
        .unwrap();
        assert_eq!(
            binding_accesses(&linear, 4),
            vec![
                BindingAccess::ReadOnly,
                BindingAccess::ReadOnly,
                BindingAccess::WriteOnly,
                BindingAccess::MayWrite,
            ]
        );
        let adam = ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(
            "../shaders/adamw.spv"
        )))
        .unwrap();
        let access = binding_accesses(&adam, 4);
        assert_eq!(access[0], BindingAccess::MayWrite);
        assert_eq!(access[1], BindingAccess::ReadOnly);
        assert_eq!(access[2], BindingAccess::MayWrite);
        assert_eq!(access[3], BindingAccess::MayWrite);
    }

    fn module(instructions: &[&[u32]]) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x10000, 0, 100, 0];
        for instruction in instructions {
            words.push((((instruction.len()) as u32) << 16) | instruction[0]);
            words.extend_from_slice(&instruction[1..]);
        }
        words
    }

    #[test]
    fn requires_all_block_members_and_merges_aliased_declarations() {
        let mut words = module(&[
            &[71, 10, 33, 0],
            &[71, 10, 34, 0],
            &[30, 20, 1, 1],
            &[32, 30, 2, 20],
            &[59, 30, 10, 2],
            &[72, 20, 0, 24],
        ]);
        assert_eq!(binding_accesses(&words, 1), vec![BindingAccess::MayWrite]);
        words.extend_from_slice(&[(4 << 16) | 72, 20, 1, 24]);
        assert_eq!(binding_accesses(&words, 1), vec![BindingAccess::ReadOnly]);
        let alias = module(&[
            &[71, 11, 33, 0],
            &[71, 11, 34, 0],
            &[30, 21, 1],
            &[32, 31, 12, 21],
            &[59, 31, 11, 12],
            &[71, 11, 25],
        ]);
        words.extend_from_slice(&alias[5..]);
        assert_eq!(binding_accesses(&words, 1), vec![BindingAccess::MayWrite]);
    }

    #[test]
    fn unknown_sets_decorations_and_malformed_words_are_conservative() {
        let words = module(&[
            &[71, 10, 33, 0],
            &[71, 10, 34, 1],
            &[71, 10, 24],
            &[32, 30, 2, 20],
            &[59, 30, 10, 2],
        ]);
        assert_eq!(binding_accesses(&words, 1), vec![BindingAccess::MayWrite]);
        assert_eq!(binding_accesses(&[], 1), vec![BindingAccess::MayWrite]);
        let mut malformed = words;
        malformed.push(0);
        assert_eq!(
            binding_accesses(&malformed, 1),
            vec![BindingAccess::MayWrite]
        );
    }
}
