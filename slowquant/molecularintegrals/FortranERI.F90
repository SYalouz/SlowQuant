
! Connect runERI to F2PY
MODULE LOL

contains

SUBROUTINE runERI(basisidx, basisfloat, basisint, max_angular_moment, E1arr, E2arr, E3arr, ERI)
    IMPLICIT NONE
    ! INPUTS
    INTEGER, INTENT(in) :: max_angular_moment
    INTEGER, DIMENSION(:,:), INTENT(in) :: basisidx, basisint
    REAL(8), DIMENSION(:,:), INTENT(in) :: basisfloat
    REAL(8), DIMENSION(:,:,:,:,:), INTENT(in) :: E1arr, E2arr, E3arr
    
    ! OUTPUTS
    REAL(8), DIMENSION(size(basisidx,1),size(basisidx,1),size(basisidx,1),size(basisidx,1)), INTENT(out) :: ERI
    
    ! INTERNAL
    INTEGER :: i, j, k, l, mu, nu, lam, sig, munu, lamsig, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, ALSTAT
    REAL(8) :: a, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, Px, Py, Pz, Qx, Qy, Qz, p, q, alpha, Normalization1, &
    &          Normalization2, Normalization3, Normalization4, c1, c2, c3, c4, calc, outputvalue
    REAL(8), DIMENSION(:), ALLOCATABLE :: E1, E2, E3, E4, E5, E6
    REAL(8), DIMENSION(:,:,:), ALLOCATABLE :: R1buffer
    REAL(8), DIMENSION(:,:,:,:), ALLOCATABLE  :: Rbuffer
    
    ! ALLOCATE INTERNALS
    ALLOCATE(E1(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E2(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E3(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E4(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E5(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E6(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(R1buffer(4*max_angular_moment+1,4*max_angular_moment+1,4*max_angular_moment+1), STAT=ALSTAT)
    ALLOCATE(Rbuffer(4*max_angular_moment+1,4*max_angular_moment+1,4*max_angular_moment+1,12*max_angular_moment+1), STAT=ALSTAT)
    !if(ALSTAT /= 0) STOP "***NEED MORE PYLONS***" 
    
	ERI = 0.0d0

    munu = 0
    lamsig = 0
    ! Loop over basisfunctions
    DO mu = 0, size(basisidx,1)-1
        DO nu = mu, size(basisidx,1)-1
            munu = mu*(mu+1)/2+nu
            DO lam = 0, size(basisidx,1)-1
                DO sig = lam, size(basisidx,1)-1
                    lamsig = lam*(lam+1)/2+sig
                    IF (munu >= lamsig) THEN
                        calc = 0.0d0
                        ! Loop over primitives
                        DO i = basisidx(mu+1,2), basisidx(mu+1,2)+basisidx(mu+1,1)-1
                            Normalization1 = basisfloat(i+1,1)
                            a  = basisfloat(i+1,2)
                            c1 = basisfloat(i+1,3)
                            Ax = basisfloat(i+1,4)
                            Ay = basisfloat(i+1,5)
                            Az = basisfloat(i+1,6)
                            l1 =   basisint(i+1,1)
                            m1 =   basisint(i+1,2)
                            n1 =   basisint(i+1,3)
                            DO j = basisidx(nu+1,2), basisidx(nu+1,2)+basisidx(nu+1,1)-1
                                Normalization2 = basisfloat(j+1,1)
                                b  = basisfloat(j+1,2)
                                c2 = basisfloat(j+1,3)
                                Bx = basisfloat(j+1,4)
                                By = basisfloat(j+1,5)
                                Bz = basisfloat(j+1,6)
                                l2 =   basisint(j+1,1)
                                m2 =   basisint(j+1,2)
                                n2 =   basisint(j+1,3)
                                p   = a+b
                                Px  = (a*Ax+b*Bx)/p
                                Py  = (a*Ay+b*By)/p
                                Pz  = (a*Az+b*Bz)/p
                                E1 = E1arr(mu+1,nu+1,i-basisidx(mu+1,2)+1,j-basisidx(nu+1,2)+1,:)
                                E2 = E2arr(mu+1,nu+1,i-basisidx(mu+1,2)+1,j-basisidx(nu+1,2)+1,:)
                                E3 = E3arr(mu+1,nu+1,i-basisidx(mu+1,2)+1,j-basisidx(nu+1,2)+1,:)
                                DO k = basisidx(lam+1,2), basisidx(lam+1,2)+basisidx(lam+1,1)-1
                                    Normalization3 = basisfloat(k+1,1)
                                    c  = basisfloat(k+1,2)
                                    c3 = basisfloat(k+1,3)
                                    Cx = basisfloat(k+1,4)
                                    Cy = basisfloat(k+1,5)
                                    Cz = basisfloat(k+1,6)
                                    l3 =   basisint(k+1,1)
                                    m3 =   basisint(k+1,2)
                                    n3 =   basisint(k+1,3)
                                    DO l = basisidx(sig+1,2), basisidx(sig+1,2)+basisidx(sig+1,1)-1
                                        Normalization4 = basisfloat(l+1,1)
                                        d  = basisfloat(l+1,2)
                                        c4 = basisfloat(l+1,3)
                                        Dx = basisfloat(l+1,4)
                                        Dy = basisfloat(l+1,5)
                                        Dz = basisfloat(l+1,6)
                                        l4 =   basisint(l+1,1)
                                        m4 =   basisint(l+1,2)
                                        n4 =   basisint(l+1,3)                                    
                                        q   = c+d
                                        Qx  = (c*Cx+d*Dx)/q
                                        Qy  = (c*Cy+d*Dy)/q
                                        Qz  = (c*Cz+d*Dz)/q
                                        E4 = E1arr(lam+1,sig+1,k-basisidx(lam+1,2)+1,l-basisidx(sig+1,2)+1,:)
                                        E5 = E2arr(lam+1,sig+1,k-basisidx(lam+1,2)+1,l-basisidx(sig+1,2)+1,:)
                                        E6 = E3arr(lam+1,sig+1,k-basisidx(lam+1,2)+1,l-basisidx(sig+1,2)+1,:)
                                        alpha = p*q/(p+q)
                                        
                                        call R(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Qx, Qy, Qz, Px,&
                                        &		Py, Pz, alpha, R1buffer, Rbuffer)
                                        call elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4,&
                                        &			Normalization1, Normalization2, Normalization3,&
                                        &			Normalization4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6,&
                                        &			R1buffer, outputvalue)
                                        calc = calc + outputvalue
                                    END DO
                                END DO
                            END DO
                        END DO
                        ERI(mu+1,nu+1,lam+1,sig+1) = calc
                        ERI(nu+1,mu+1,lam+1,sig+1) = calc
                        ERI(mu+1,nu+1,sig+1,lam+1) = calc
                        ERI(nu+1,mu+1,sig+1,lam+1) = calc
                        ERI(lam+1,sig+1,mu+1,nu+1) = calc
                        ERI(sig+1,lam+1,mu+1,nu+1) = calc
                        ERI(lam+1,sig+1,nu+1,mu+1) = calc
                        ERI(sig+1,lam+1,nu+1,mu+1) = calc
                    END IF
                END DO
            END DO
        END DO
    END DO
    !return ERI
END SUBROUTINE runERI
        
PURE SUBROUTINE R(l1l2, m1m2, n1n2, Cx, Cy, Cz, Px, Py, Pz, p, R1, Rbuffer)
    IMPLICIT NONE
    !EXTERNAL Boys_func
    
    ! INPUTS
    INTEGER, INTENT(in) :: l1l2, m1m2, n1n2
    REAL(8) , INTENT(in):: Cx, Cy, Cz, Px, Py, Pz, p
    REAL(8), DIMENSION(:,:,:,:), INTENT(inout) :: Rbuffer
    
    ! OUTPUTS
    REAL(8), DIMENSION(:,:,:), INTENT(out) :: R1
    
    ! INTERNAL
    INTEGER :: t, u, v, n, exclude_from_n
    REAL(8) :: RPC, PCx, PCy, PCz, outputvalue, F
    
    PCx = Px-Cx
    PCy = Py-Cy
    PCz = Pz-Cz
    RPC = ((PCx)**2+(PCy)**2+(PCz)**2)**0.5
    DO t = 0, l1l2
        DO u = 0, m1m2
            DO v = 0, n1n2
                ! Check the range of n, to ensure no redundent n are calculated
                IF (t == 0 .AND. u == 0) THEN
                    exclude_from_n = v
                ELSEIF (t == 0) THEN
                    exclude_from_n = n1n2 + u
                ELSE
                    exclude_from_n = n1n2 + m1m2 + t
                END IF
                
                DO n = 0, l1l2+m1m2+n1n2 - exclude_from_n
                    outputvalue = 0.0d0
                    IF (t == 0 .AND. u == 0 .AND. v == 0) THEN
						CALL boys(real(n,8),p*RPC*RPC,F)
                        Rbuffer(t+1,u+1,v+1,n+1) = (-2.0d0*p)**n*F
                    ELSE
                        IF (t == 0 .AND. u == 0) THEN
                            IF (v > 1) THEN
                                outputvalue = outputvalue + (v-1.0d0)*Rbuffer(t+1,u+1,v+1-2,n+1+1)
                            END IF
                            outputvalue = outputvalue + PCz*Rbuffer(t+1,u+1,v+1-1,n+1+1)  
                        ELSEIF (t == 0) THEN
                            IF (u > 1) THEN
                                outputvalue = outputvalue + (u-1.0d0)*Rbuffer(t+1,u+1-2,v+1,n+1+1)
                            END IF
                            outputvalue = outputvalue + PCy*Rbuffer(t+1,u+1-1,v+1,n+1+1)
                        ELSE
                            IF (t > 1) THEN
                                outputvalue = outputvalue + (t-1.0d0)*Rbuffer(t+1-2,u+1,v+1,n+1+1)
                            END IF
                            outputvalue = outputvalue + PCx*Rbuffer(t+1-1,u+1,v+1,n+1+1)
                        END IF
                        Rbuffer(t+1,u+1,v+1,n+1) = outputvalue
                    END IF
                        
                    IF (n == 0) THEN
                        R1(t+1,u+1,v+1) = Rbuffer(t+1,u+1,v+1,n+1)
                    END IF
                END DO
            END DO
        END DO
    END DO
    !return R1
END SUBROUTINE R


PURE SUBROUTINE elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Normalization1, Normalization2, &
&                        Normalization3, Normalization4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6, Rpre, outputvalue)
    IMPLICIT NONE
    ! INPUT
    INTEGER, INTENT(in) :: l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4
    REAL(8), INTENT(in) :: p, q, Normalization1, Normalization2, Normalization3, Normalization4, c1, c2, c3, c4
    REAL(8), DIMENSION(:), INTENT(in) :: E1, E2, E3, E4, E5, E6
    REAL(8), DIMENSION(:,:,:), INTENT(in) :: Rpre
    
    ! OUTPUT
    REAL(8), INTENT(out) :: outputvalue
    
    ! INTERNAL
    INTEGER :: tau, nu, phi, t, u, v
    REAL(8) :: N, factor
    REAL(8), PARAMETER :: pi = 3.141592653589793238462643383279d0
                        
    N = Normalization1*Normalization2*Normalization3*Normalization4*c1*c2*c3*c4
    outputvalue = 0.0d0
    DO tau = 0, l3+l4
        DO nu = 0, m3+m4
            DO phi = 0, n3+n4
                factor = (-1.0d0)**(tau+nu+phi)
                DO t = 0, l1+l2
                    DO u = 0, m1+m2
                        DO v = 0, n1+n2
                            outputvalue = outputvalue + E1(t+1)*E2(u+1)*E3(v+1)*&
                            &             E4(tau+1)*E5(nu+1)*E6(phi+1)*Rpre(t+tau+1,u+nu+1,v+phi+1)*factor
                        END DO
                    END DO
                END DO
            END DO
        END DO
    END DO
    outputvalue = outputvalue*2.0d0*pi**2.5/(p*q*(p+q)**0.5)*N
END SUBROUTINE elelrep

PURE SUBROUTINE factorial2(n, outval)
	IMPLICIT NONE
	REAL(8), INTENT(in) :: n
	INTEGER :: n_range, i
	REAL(8), INTENT(out) :: outval
	n_range = int(n)
	outval = 1.0d0
	IF (n > 0) THEN
		DO i=0, (n_range+1)/2-1
			outval = outval*(n-2*i)
		END DO
	END IF
END SUBROUTINE factorial2

PURE SUBROUTINE boys(m, z, F)
	IMPLICIT NONE
	REAL(8), INTENT(in) :: m, z
	INTEGER :: i
	REAL(8) :: Fcheck, outval, temp1
	REAL(8), PARAMETER :: pi = 3.141592653589793238462643383279
	REAL(8), INTENT(out) :: F
	IF (z > 25.0d0) THEN
	    ! long range approximation
		CALL factorial2(2*m-1,outval)
		F = outval/(2.0d0**(m+1))*(pi/(z**(2*m+1)))**0.5
	ElSE IF (z == 0) THEN
		! special case of T = 0
        F = 1.0d0/(2.0d0*m+1.0d0)
	ELSE
		F = 0.0d0
		CALL factorial2(2*m-1,outval)
		temp1 = outval
		DO i=0, 100
			Fcheck = F
			CALL factorial2(2*m+2*i+1,outval)
			F = F + (temp1*(2.0d0*z)**i)/outval
			Fcheck = Fcheck - F
			! threshold from purple book
			IF (ABS(Fcheck) < 10e-10) THEN
				EXIT
			END IF
		END DO
		F = F*exp(-z)
	END IF
END SUBROUTINE boys


END MODULE LOL


