"""
变量索引映射工具。

管理 (element, bit_position) ↔ [0..37] flat index 的双向映射。
"""

from typing import Dict, List, Tuple

from ..physical_params import MAIN_ELEMENTS, INTERSTITIAL_ELEMENT, ENCODING


class VariableMapper:
    """38 变量索引的双向映射器。

    变量布局:
      x_0  ~ x_6   → Al: b_0 ~ b_6
      x_7  ~ x_13  → Co: b_0 ~ b_6
      x_14 ~ x_20  → Cr: b_0 ~ b_6
      x_21 ~ x_27  → Fe: b_0 ~ b_6
      x_28 ~ x_34  → Ni: b_0 ~ b_6
      x_35 ~ x_37  → C:  b_0 ~ b_2
    """

    def __init__(self):
        self._params = ENCODING
        self._flat_to_elem: Dict[int, Tuple[str, int]] = {}
        self._elem_to_start: Dict[str, Tuple[int, int]] = {}  # (start, n_bits)

        idx = 0
        for elem in MAIN_ELEMENTS:
            start = idx
            for bit_pos in range(self._params.bits_main):
                self._flat_to_elem[idx] = (elem, bit_pos)
                idx += 1
            self._elem_to_start[elem] = (start, self._params.bits_main)

        # C
        start = idx
        for bit_pos in range(self._params.bits_carbon):
            self._flat_to_elem[idx] = (INTERSTITIAL_ELEMENT, bit_pos)
            idx += 1
        self._elem_to_start[INTERSTITIAL_ELEMENT] = (
            start,
            self._params.bits_carbon,
        )

    @property
    def total_variables(self) -> int:
        return self._params.total_variables

    def flat_to_element_bit(self, flat_idx: int) -> Tuple[str, int]:
        """flat index → (element_name, bit_position)。"""
        return self._flat_to_elem[flat_idx]

    def element_bit_to_flat(self, elem: str, bit_pos: int) -> int:
        """(element_name, bit_position) → flat index。"""
        start, n_bits = self._elem_to_start[elem]
        if not (0 <= bit_pos < n_bits):
            raise ValueError(
                f"Bit position {bit_pos} out of range [0, {n_bits}) for {elem}"
            )
        return start + bit_pos

    def get_element_slice(self, elem: str) -> slice:
        """返回某元素在 flat 数组中的切片。"""
        start, n_bits = self._elem_to_start[elem]
        return slice(start, start + n_bits)

    def get_element_indices(self, elem: str) -> List[int]:
        """返回某元素所有 flat index。"""
        start, n_bits = self._elem_to_start[elem]
        return list(range(start, start + n_bits))

    def get_main_element_indices(self) -> List[int]:
        """返回所有主元 flat indices (0~34)。"""
        result = []
        for elem in MAIN_ELEMENTS:
            result.extend(self.get_element_indices(elem))
        return result

    def get_carbon_indices(self) -> List[int]:
        """返回 C 元素 flat indices (35~37)。"""
        return self.get_element_indices(INTERSTITIAL_ELEMENT)

    def get_cr_indices(self) -> List[int]:
        """返回 Cr 元素 flat indices (14~20), P2 约束专用。"""
        return self.get_element_indices("Cr")

    @property
    def cr_start(self) -> int:
        return self._elem_to_start["Cr"][0]

    @property
    def carbon_start(self) -> int:
        return self._elem_to_start[INTERSTITIAL_ELEMENT][0]

    def get_all_variable_names(self) -> List[str]:
        """返回所有变量名列表 ['x_0', 'x_1', ..., 'x_37']."""
        return [f"x_{i}" for i in range(self.total_variables)]


# GLOBAL singleton
_VARIABLE_MAPPER: VariableMapper = None


def get_variable_mapper() -> VariableMapper:
    global _VARIABLE_MAPPER
    if _VARIABLE_MAPPER is None:
        _VARIABLE_MAPPER = VariableMapper()
    return _VARIABLE_MAPPER
