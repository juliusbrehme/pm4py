'''
PM4Py – A Process Mining Library for Python
Copyright (C) 2026 Process Intelligence Solutions GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
'''
import sys
from pm4py.objects.process_tree.obj import ProcessTree, Operator


class EnhancedProcessTree(ProcessTree):
    """
    Process tree whose nodes carry additional start / stop / skip annotations.

    - start: execution of the overall process may begin at this node
    - stop:  execution of the overall process may terminate after this node
    - skip:  this node may be bypassed within its parent
    """

    def __init__(self, operator=None, parent=None, children=None, label=None,
                 start=False, stop=False, skip=False):
        super().__init__(operator=operator, parent=parent,
                         children=children, label=label)
        self._start = start
        self._stop = stop
        self._skip = skip

    def _get_start(self):
        return self._start

    def _set_start(self, value):
        self._start = bool(value)

    def _get_stop(self):
        return self._stop

    def _set_stop(self, value):
        self._stop = bool(value)

    def _get_skip(self):
        return self._skip

    def _set_skip(self, value):
        self._skip = bool(value)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if isinstance(other, EnhancedProcessTree):
            return (self._start == other.start
                    and self._stop == other.stop
                    and self._skip == other.skip)
        return True

    # annotations deliberately excluded: must stay consistent with __eq__
    # against plain ProcessTree instances
    def __hash__(self):
        return super().__hash__()

    def to_string(self, level=0, indent=False, max_indent=sys.maxsize):
        start_indicator = " (start)" if self.start else ""
        stop_indicator = " (stop)" if self.stop else ""
        skip_indicator = " (skip)" if self.skip else ""

        if self.label is not None:
            return f"{start_indicator}'{self.label}'{stop_indicator}{skip_indicator}"

        if self.operator is None and self.label is None:
            return "tau"

        children_strs = [
            child.to_string(level=level + 1, indent=indent, max_indent=max_indent)
            for child in self.children
        ]

        if indent and level < max_indent:
            indent_str = "\n" + "\t" * (level + 1)
            children_joined = indent_str + indent_str.join(children_strs)
            return f"{start_indicator}{self.operator}{stop_indicator}{skip_indicator}( {children_joined} \n{'\t' * level})"
        else:
            children_joined = ", ".join(children_strs)
            return f"{start_indicator}{self.operator}{stop_indicator}{skip_indicator}( {children_joined} )"

    start = property(_get_start, _set_start)
    stop = property(_get_stop, _set_stop)
    skip = property(_get_skip, _set_skip)