# Neural ALU bit encoding

Each operation uses **40 input bits (5 bytes)**, in this order:

```text
| Left operand: 16 bits | Right operand: 16 bits | Operation type: 8 bits |
|         0–15         |          16–31         |          32–39         |
```

Positions are zero-based, counted from left to right. Each field stores its
most significant bit first. One operation byte contains **8 bits**, so keys
include leading zeros, such as `00000001`.

## Operands

For now, the supported input types are **`bool` and `int` only**. Floats and
strings are unsupported, including whole-valued floats such as `1.0` and
single-character strings such as `"A"`. Integers must fit the configured signed
operand width.

The first 16 bits encode the left side of the operation, and the next 16 bits
encode the right side. For `left - right`, their order matters.

Both operands use signed two's-complement integers, from **−32,768 to 32,767**:

| Value | 16-bit encoding |
| --- | --- |
| 0 | `0000000000000000` |
| 1 | `0000000000000001` |
| 2 | `0000000000000010` |
| −1 | `1111111111111111` |
| −32,768 | `1000000000000000` |
| 32,767 | `0111111111111111` |

Boolean inputs use 0 for false and 1 for true. Unary operations use the left
operand and set all right-operand bits to zero. Fractional inputs and integers
outside this range are not supported by the initial codec.

### Should we add `char`?

Recommendation: defer a distinct character type. Python represents characters
as one-character strings, and this bit format has no input type tags. For
example, the bits for integer `65` cannot also identify a distinct character
`"A"` without an additional convention.

An application can explicitly map a limited character set to integers before
calling the ALU, then map results back outside it. That is integer processing,
not native character support. Future character support needs a defined encoding,
supported range, operation semantics, and type metadata or dedicated opcodes.
Do not assume one 16-bit field can represent every Unicode character.

## Operation types

The final byte selects the operation. These keys preserve the existing
`OPCODE_IDS` assignments in `allmengine/src/allmengine/network.py`.

| 8-bit key | Decimal ID | Operation type | Meaning |
| --- | --- | --- | --- |
| `00000000` | 0 | `add` | Addition: `left + right` |
| `00000001` | 1 | `sub` | Subtraction: `left - right` |
| `00000010` | 2 | `mul` | Multiplication: `left * right` |
| `00000011` | 3 | `truediv` | True division: `left / right` |
| `00000100` | 4 | `floordiv` | Floor division: `left // right` |
| `00000101` | 5 | `mod` | Remainder: `left % right` |
| `00000110` | 6 | `pow` | Exponentiation: `left ** right` |
| `00000111` | 7 | `matmul` | Matrix multiplication: `left @ right` |
| `00001000` | 8 | `lshift` | Left shift: `left << right` |
| `00001001` | 9 | `rshift` | Right shift: `left >> right` |
| `00001010` | 10 | `and_` | Bitwise AND: `left & right` |
| `00001011` | 11 | `or_` | Bitwise OR: `left \| right` |
| `00001100` | 12 | `xor` | Bitwise XOR: `left ^ right` |
| `00001101` | 13 | `lt` | Less than: `left < right` |
| `00001110` | 14 | `le` | Less than or equal: `left <= right` |
| `00001111` | 15 | `eq` | Equal: `left == right` |
| `00010000` | 16 | `ne` | Not equal: `left != right` |
| `00010001` | 17 | `gt` | Greater than: `left > right` |
| `00010010` | 18 | `ge` | Greater than or equal: `left >= right` |
| `00010011` | 19 | `neg` | Unary negation: `-left` |
| `00010100` | 20 | `pos` | Unary positive: `+left` |
| `00010101` | 21 | `abs` | Absolute value: `abs(left)` |
| `00010110` | 22 | `invert` | Bitwise inversion: `~left` |
| `00010111` | 23 | `bool` | Truthiness: `bool(left)` |

Keys `00011000` through `11111111` (24–255) are unassigned. Assign new operations
unused keys without renumbering existing ones.

`and_` and `or_` are bitwise operations. Python's short-circuit `and`, `or`, and
`not` use the truthiness hook (`bool`); they do not have separate opcodes.

An assigned key identifies the intended operation, not a guarantee that the
untrained network computes it correctly. Matrix operands and fractional division
results cannot be represented by the current signed-integer codec.

## Result encoding

The network emits **16 output bits (2 bytes)**, most significant bit first.
There is no output type tag. The caller knows the opcode and uses it to choose
the decoder; **the result type depends on the operation, not just the input
types**. For example, two integers can produce either an integer (`3 + 4`) or
a boolean (`3 < 4`).

| Operation category | Decoded type | Interpretation |
| --- | --- | --- |
| Comparisons: `lt`, `le`, `eq`, `ne`, `gt`, `ge` | `bool` | True if more than 8 output bits are 1 |
| Truthiness: `bool` | `bool` | True if more than 8 output bits are 1 |
| All other assigned operations | `int` | Read all 16 bits as signed two's complement |

Boolean and integer inputs share an encoding: `True` and `1` have identical
bits, as do `False` and `0`. The network cannot recover their original Python
types. Even bitwise operations on boolean inputs currently decode as integers.
There is no `char` result type yet.

### Integer results

Use the same signed two's-complement representation as the operands, with a
range of **-32,768 to 32,767**. If the unsigned value of the 16 bits is `u`,
decode it as `u` when the first bit is zero, otherwise as `u - 65536`.

| Intended result | Output bits |
| --- | --- |
| `3 + 4` = `7` | `0000000000000111` |
| `3 - 4` = `-1` | `1111111111111111` |
| `6 & 3` = `2` | `0000000000000010` |

There is no overflow flag, fractional representation, or exception flag in the
output. Results outside the signed range cannot be represented exactly. The
current decoder always returns an integer for arithmetic opcodes, including
`truediv`; this does not implement Python's floating-point division semantics.

### Boolean results

Boolean results use **strict majority voting across all 16 output bits**:

- **9–16 bits set to 1:** `True`.
- **0–8 bits set to 1:** `False`.
- An 8-to-8 tie is `False`, not an error or an uncertain result.

The decoding rule is `sum(output_bits) > 8`. Bit positions do not affect the
vote. For a configurable output width, use `sum(output_bits) > len(output_bits) / 2`.

Canonical encodings for training targets:

| Result | Canonical output bits |
| --- | --- |
| `False` | `0000000000000000` |
| `True` | `1111111111111111` |

| Example output | Number of ones | Boolean result |
| --- | --- | --- |
| `0000000000000001` | 1 | `False` |
| `1111111100000000` | 8 | `False` (tie) |
| `1111111110000000` | 9 | `True` |
| `1111111111111110` | 15 | `True` |

This replaces the previous single-output-bit convention. Boolean **inputs**
still encode false as integer 0 and true as integer 1; majority voting applies
only to boolean **outputs**. Integer outputs still use signed two's complement.

The intended behavior of the `bool` operation is still Python integer
truthiness: zero is false and every nonzero integer is true. The network must
learn that mapping from the input to the output majority; the decoder
does not compute truthiness from the input itself.

These examples describe desired outputs. Random, untrained weights do not
guarantee correct arithmetic or boolean results.

## Input examples

`3 + 4` uses the addition key `00000000`:

```text
0000000000000011 0000000000000100 00000000
left = 3         right = 4        add
```

`6 & 3` uses the bitwise AND key `00001010`:

```text
0000000000000110 0000000000000011 00001010
left = 6         right = 3        and_
```

Spaces separate fields for readability; the network receives 40 individual bits.

## Implementation alignment

`BinaryNetwork.forward` implements the majority-vote rule for comparisons and
truthiness. The threshold follows the configured output width; ties are false.
Network checkpoints have no version requirement. Boolean decoding always uses
majority voting, including when loading older weights. Any legacy version field
is ignored and is not written into new checkpoints.

This document specifies the requested **left → right → operation** layout.
The current `BinaryNetwork.encode` implementation still emits
**operation → left → right**. Its field order must be updated before treating
these examples as its actual encoded output. Existing trained weights would
also need input-column migration or retraining after that change.

The widths above describe the default configuration in
`allmengine/src/allmengine/network.json`. Other configurations retain the same
field meanings, with total width equal to `2 * operand_bits + opcode_bits`.
