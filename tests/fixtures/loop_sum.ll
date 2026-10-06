; ModuleID = 'loop_sum'
; A function with a counted loop summing array elements.
; Expected: 11 instructions, 3 BBs (entry, loop, exit), 1 loop depth 1,
;           2 phi nodes, 1 load, 1 GEP, 2 int_arith, 1 icmp, 2 cond/uncond br.

define i32 @loop_sum(ptr %arr, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr inbounds i32, ptr %arr, i32 %i
  %val = load i32, ptr %ptr
  %sum.next = add nsw i32 %sum, %val
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}
