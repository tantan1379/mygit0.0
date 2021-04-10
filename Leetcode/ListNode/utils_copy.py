class copy:
    def FindMid(self,head):
        if(not head and not head.next):
            return head
        pf = head
        ps = head
        while(pf.next and pf.next.next):
            pf = pf.next.next
            ps = ps.next
        return ps

    def ReverseLinkList(self,head):
        prev = None
        cur = head
        while(cur):
            pnext = cur.next
            cur.next = prev
            prev = cur
            cur = pnext
        return prev