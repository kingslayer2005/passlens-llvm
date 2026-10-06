; ModuleID = 'branch_heavy'
; A function with many conditional branches and no loops.
; Expected: 13 instructions, 5 BBs, 0 loops, 4 cond_br (icmp+br pairs),
;           1 uncond_br, 3 icmp, multiple CFG edges.

define i32 @branch_heavy(i32 %x) {
entry:
  %c1 = icmp sgt i32 %x, 10
  br i1 %c1, label %then1, label %else1

then1:
  %a = add nsw i32 %x, 1
  %c2 = icmp sgt i32 %a, 20
  br i1 %c2, label %then2, label %merge

else1:
  %b = sub nsw i32 %x, 1
  %c3 = icmp slt i32 %b, 0
  br i1 %c3, label %then2, label %merge

then2:
  %d = mul nsw i32 %x, 2
  br label %merge

merge:
  %result = phi i32 [ %a, %then1 ], [ %b, %else1 ], [ %d, %then2 ]
  ret i32 %result
}
