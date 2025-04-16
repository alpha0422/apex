"""Scheduling abstractions on PyTorch Inductor WrapperCodeGen level.

Attributes:
    DEFAULT_STREAM: Name of the default CUDA Stream on the final generated Python code.
    DEFAULT_STREAM_IDX: Index number of the default CUDA Stream in `torchsched` internal passes.
    STREAM_NAME_TEMPLATE: Python string template to generate stream names. Can be used as:

            idx: int = ...
            stream = STREAM_NAME_TEMPLATE.format(stream_idx=idx)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from torch._inductor.codegen.wrapper import BufferLike
from torch._inductor.codegen.wrapper import EnterDeviceContextManagerLine
from torch._inductor.codegen.wrapper import ExitDeviceContextManagerLine
from torch._inductor.codegen.wrapper import IndentedBuffer
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
from torch._inductor.codegen.wrapper import WrapperLine
from torch._inductor.virtualized import V

import apex.contrib.torchsched.config as config
from apex.contrib.torchsched.inductor._utils import DEFAULT_STREAM
from apex.contrib.torchsched.inductor._utils import DEFAULT_STREAM_IDX
from apex.contrib.torchsched.inductor._utils import ENTRANCE_EVENT
from apex.contrib.torchsched.inductor._utils import STREAM_NAME_TEMPLATE
from apex.contrib.torchsched.inductor._utils import get_stream_name

if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import GraphPartitionSignature

    from apex.contrib.torchsched.inductor.event import CudaEventSym


@dataclasses.dataclass
class EnterDeviceContextManagerWithStreamInfoLine(EnterDeviceContextManagerLine):
    """Enter a CUDA device context and allocate required side streams.

    Note:
        - The number of allocated streams is controlled by :attr:`apex.contrib.torchsched.config.num_streams`;
    """

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate context switching and stream allocation code."""
        if V.graph.cpp_wrapper:
            super().codegen(code)
        else:
            super().codegen(code)
            code.writeline(f"{DEFAULT_STREAM} = torch.cuda.current_stream()")
            code.writeline(f"{ENTRANCE_EVENT} = {DEFAULT_STREAM}.record_event()")

            code.writeline("from apex.contrib.torchsched.inductor._utils import get_cuda_stream_pool")
            code.writeline(
                f"cuda_stream_pool = get_cuda_stream_pool(device={self.device_idx}, "
                f"pool_size={config.num_streams})",
            )

            for i in range(1, config.num_streams):
                code.writeline(
                    f"{STREAM_NAME_TEMPLATE.format(stream_idx=i)} "
                    f"= cuda_stream_pool.acquire()",
                )

@dataclasses.dataclass
class ExitDeviceContextManagerWithStreamInfoLine(ExitDeviceContextManagerLine):
    """Exit a CUDA device context and release allocated streams."""

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate context switching and stream release code."""
        for i in range(1, config.num_streams):
            code.writeline(
                f"cuda_stream_pool.release({STREAM_NAME_TEMPLATE.format(stream_idx=i)})",
            )
        if not V.graph.cpp_wrapper:
            code.do_unindent()


@dataclasses.dataclass
class EnterCudaStreamContextLine(WrapperLine):
    """Enter a context executed by respective CUDA Stream and insert necessary syncs.

    Attributes:
        wrapper: The code-gen wrapper of the current compilation phase.
        stream_idx: The index number corresponds to the entering CUDA Stream context.
        upstream_events: Names of CUDA Events that the current stream should be waiting for before
            the stream switching.
    """

    wrapper: MultiStreamWrapperCodegen
    stream_idx: int
    upstream_events: set[CudaEventSym]

    def __post_init__(self) -> None:
        """Construct stream name by the given index number."""
        self.stream_name = get_stream_name(self.stream_idx)
        # Symbolic event dependency should be handled in the 1st pass of code-gen so the reference
        # counters can be properly tracked.
        for event in self.upstream_events:
            self.wrapper.writeline(event.wait(self.stream_idx))

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate event sync and stream switching code."""
        code.writeline(f"with torch.cuda.stream({self.stream_name}):")
        code.do_indent()

        # [NOTE] The 3-indent-level assertion
        #
        #     Indent level 1: Inductor wrapper call indent
        #         Indent level 2: Device guard context indent
        #             Indent level 3: CUDA Stream context indent
        #
        # Over or under indenting usually means that :meth:`MultiCudaStreamScheduler.codegen`
        # introduced bugs on stream context switching. This check also applies to stream context
        # exiting, as in :meth:`ExitCudaStreamContextLine.codegen`.
        assert code._indent == 3


@dataclasses.dataclass
class ExitCudaStreamContextLine(WrapperLine):
    """Generate code to exit the current stream context.

    Attributes:
        stream_idx: The index number corresponds to the entering CUDA Stream context.
        buffers_from_other_streams: Name of buffers produced by other CUDA Streams. Those buffers
            should be recorded to the current stream to avoid accidental memory free.
        buffers_requiring_device_check: Name of buffers that might not be on CUDA devices and
            require runtime device checking before recording stream to them.

    Note:
        Most attributes and checking logics of this class have been moved to
        :meth:`MultiStreamWrapperCodeGen.codegen_cuda_stream_exit`. We preserve this data structure
        because the checking and unindent should be generated in the latter phase of code-gen.
    """

    stream_idx: int
    buffers_from_other_streams: set[str]
    buffers_requiring_device_check: set[str]

    def __post_init__(self) -> None:
        """Construct stream name by the given index number."""
        self.stream_name = get_stream_name(self.stream_idx)

    def codegen(self, code: IndentedBuffer) -> None:
        """Check indentation level and exit the current stream context."""
        assert code._indent == 3  # See :note:`The 3-indent-level assertion` above.

        # Record the buffer created by another stream has been used by this stream, thus to avoid
        # the buffer being freed before its usage on this stream.
        for buff in self.buffers_from_other_streams:
            prefix = f"if {buff}.is_cuda: " if buff in self.buffers_requiring_device_check else ""
            code.writeline(f"{prefix}{buff}.record_stream({self.stream_name})")

        code.do_unindent()


@dataclasses.dataclass
class RecordBuffFromSideStreams(WrapperLine):
    """Record the returning tensors on the default stream to avoid accidental memory free.

    Attributes:
        buffers: Names of the returning tensors that need to be recorded on the default stream.
        buffers_requiring_device_check: Name of buffers that might not be on CUDA devices and
            require runtime device checking before recording stream to them.
    """

    buffers: set[str]
    buffers_requiring_device_check: set[str]

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate stream recording for returning tensors."""
        for buff in self.buffers:
            prefix = f"if {buff}.is_cuda: " if buff in self.buffers_requiring_device_check else ""
            code.writeline(f"{prefix}{buff}.record_stream({DEFAULT_STREAM})")


class MultiStreamWrapperCodegen(PythonWrapperCodegen):
    """Wrapper code generator for graph scheduling."""

    def __init__(self) -> None:
        """Construct a code-gen wrapper and disable raw stream caching.

        Note:
            The :meth:`write_get_raw_stream` method processed in this constructor is invoked from
            literally everywhere throughout the Inductor stack, but the current
            :meth:`PythonWrapperCodegen.write_get_raw_stream` is LRU-cached and always returns a
            const raw stream name. This is not what we wanted in a multi-stream environment. Thus
            we need to re-patch this function in instance initialization.
        """
        super().__init__()
        self.current_stream_name: str | None = None
        self.write_get_raw_stream = self._write_get_raw_stream

        # List of set of buffers created by other streams
        self.ctx_buffers_from_other_streams: list[set[str]] = []
        self.ctx_buffers_requiring_device_check: list[set[str]] = []

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str,
        parent_wrapper: MultiStreamWrapperCodegen,
        partition_signatures: GraphPartitionSignature | None = None,
    ) -> MultiStreamWrapperCodegen | SubgraphPythonWrapperCodegen:
        """Instantiate a wrapper codegen for an Inductor graph or a subgraph."""
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name,
                parent_wrapper,
                partition_signatures,
            )
        return MultiStreamWrapperCodegen()

    def _write_get_raw_stream(self, device_idx: int, graph: GraphLowering | None = None) -> str:
        self.write_triton_header_once()
        if self.current_stream_name is not None:
            name = f"{self.current_stream_name}_raw"
            self.writeline(f"{name} = {self.current_stream_name}.cuda_stream")
        else:
            name = f"stream{device_idx}"
            self.writeline(f"{name} = get_raw_stream({device_idx})")
        return name

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        """Generate data structure for device guard context.

        Note:
            Refer to :class:`EnterDeviceContextManagerWithStreamInfoLine` doc for more details.
        """
        self.writeline(
            EnterDeviceContextManagerWithStreamInfoLine(
                device_idx,
                self.last_seen_device_guard_index,
            ),
        )
        self.last_seen_device_guard_index: int = device_idx

    def codegen_device_guard_exit(self) -> None:
        """Generate data structure for exiting device guard context."""
        self.writeline(ExitDeviceContextManagerWithStreamInfoLine())

    def codegen_cuda_stream_enter(
        self,
        stream_idx: int,
        upstream_events: set[CudaEventSym],
        buffers_from_other_streams: set[str],
        buffers_requiring_device_check: set[str],
    ) -> EnterCudaStreamContextLine:
        """Generate data structure for entering a CUDA Stream context.

        Note:
            - Refer to :class:`EnterCudaStreamContextLine` for argument specifications;
            - Once entered a context, the stream associated with this context will also be recorded
              such that kernels in subsequent code-gen can get the correct stream index.

        Raises:
            ValueError: If this function is called while the previous stream context isn't exited.
        """
        if self.current_stream_name is not None:
            raise ValueError(
                f"Nested stream context switching: {self.current_stream_name} -> "
                f"{get_stream_name(stream_idx)}",
            )
        ctx_entrance = EnterCudaStreamContextLine(
            self,
            stream_idx,
            upstream_events,
        )
        self.writeline(ctx_entrance)
        self.current_stream_name = ctx_entrance.stream_name

        # Within this stream's context, we need to `record_stream` for the buffers from
        # other streams:
        # 1. If such buffer is freed after its last usage, or;
        # 2. On stream exit;
        self.ctx_buffers_from_other_streams.append(buffers_from_other_streams)
        self.ctx_buffers_requiring_device_check.append(buffers_requiring_device_check)

        return ctx_entrance

    def codegen_cuda_stream_exit(
        self,
        stream_idx: int,
    ) -> None:
        """Generate data structure for exiting a CUDA Stream context.

        Args:
            stream_idx: Index of the stream context to exit from.

        Raises:
            ValueError: If attempting to exit from a different context than `steam_idx` or to exit
                from code not in any stream context.
        """
        if self.current_stream_name is None:
            raise ValueError("Attempting to exit from code that isn't in a stream context")
        elif (stream_name := get_stream_name(stream_idx)) != self.current_stream_name:
            raise ValueError(
                f"Attempting to exit from {stream_name} but the current stream context is "
                f"{self.current_stream_name}",
            )
        buffers_from_other_streams = self.ctx_buffers_from_other_streams.pop()
        buffers_requiring_device_check = self.ctx_buffers_requiring_device_check.pop()
        ctx_exit = ExitCudaStreamContextLine(
            stream_idx,
            buffers_from_other_streams,
            buffers_requiring_device_check,
        )
        self.writeline(ctx_exit)
        self.current_stream_name = None

    def make_buffer_free(self, buffer: BufferLike | ir.TorchBindObject) -> str:
        """Generate the code to free the given buffer.

        If the buffer is on side-stream, then generate record_stream too.

        Args:
            buffer (BufferLike | ir.TorchBindObject): The buffer to be freed.

        Returns:
            str: Code string to free the buffer.
        """
        line = ""
        buff = buffer.get_name()
        if self.ctx_buffers_from_other_streams and buff in self.ctx_buffers_from_other_streams[-1]:
            prefix = (
                f"if {buff}.is_cuda: "
                if buff in self.ctx_buffers_requiring_device_check[-1]
                else ""
            )
            line = f"{prefix}{buff}.record_stream({self.current_stream_name})"
            # This buffer will be freed, we shouldn't record_stream again on stream exit
            self.ctx_buffers_from_other_streams[-1].remove(buff)
        if line:
            line += f"; del {buff}"
        return line

    def codegen_sync_unjoined_cuda_events(self, events: set[CudaEventSym]) -> None:
        """Generate data structure for syncing hanging CUDA Events before program exit.

        Args:
            events: Symbols of the events that need to be waited for before program exit.
        """
        for event in events:
            self.writeline(event.wait(DEFAULT_STREAM_IDX))

    def codegen_record_buff_from_side_streams(
        self,
        buffers: set[str],
        buffers_requiring_device_check: set[str],
    ) -> None:
        """Generate data structure for recording steam on return tensors before program exit.

        Note:
            Refer to :class:`RecordBuffFromSideStreams` for more details.
        """
        self.writeline(RecordBuffFromSideStreams(buffers, buffers_requiring_device_check))
