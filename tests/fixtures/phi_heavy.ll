; ModuleID = 'phi_heavy'
; A function with many phi nodes across multiple merge points.
; Expected: 17 instructions, 6 BBs, 0 loops, 6 phi nodes total,
;           phi_args_total = 10 (1+1+2+2+2+2).

define i32 @phi_heavy(i32 %x, i32 %y) {
entry:
  %c1 = icmp sgt i32 %x, 0
  br i1 %c1, label %left, label %right

left:
  %a = add i32 %x, 1
  %c2 = icmp sgt i32 %a, 5
  br i1 %c2, label %mid_left, label %mid_right

right:
  %b = sub i32 %y, 1
  br label %mid_right

mid_left:
  %p1 = phi i32 [ %a, %left ]
  %p2 = phi i32 [ %x, %left ]
  br label %exit

mid_right:
  %p3 = phi i32 [ %a, %left ], [ %b, %right ]
  %p4 = phi i32 [ %y, %left ], [ %x, %right ]
  br label %exit

exit:
  %final = phi i32 [ %p1, %mid_left ], [ %p3, %mid_right ]
  %extra = phi i32 [ %p2, %mid_left ], [ %p4, %mid_right ]
  %result = add i32 %final, %extra
  ret i32 %result
}
